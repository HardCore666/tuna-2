# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Time-ordered interleaved image-text dataset for Tuna-2.

The dataset follows the Show-o2 mixed-modality idea: an ordered story/context is
serialized as alternating text and image spans. During training we retain an
early prefix as clean context and train the model to predict later text tokens
and denoise later images.

Supported JSONL shapes:

    {"sequence": [
      {"type": "text", "text": "A child walks into the room."},
      {"type": "image", "path": "story/0001.jpg"},
      {"type": "text", "text": "She picks up a toy."},
      {"type": "image", "path": "story/0002.jpg"}
    ]}

or a compact VIST-style shape:

    {"images": ["0001.jpg", "0002.jpg"],
     "captions": ["A child walks into the room.", "She picks up a toy."]}

Items may include explicit ``target`` booleans. If omitted, the dataset samples
a temporal split: with ``noise_all_images_prob`` probability all images are
targets; otherwise an early image prefix is clean context and the remaining
future image/text pairs are targets.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from tuna.data.tokenize_utils import format_temporal_interleaved_sequence
from tuna.data.transforms import build_image_transform


logger = logging.getLogger(__name__)


_BOI_TOKEN = "<|img_start|>"
_EOI_TOKEN = "<|img_end|>"
_IMG_PAD_TOKEN = "<|img_pad|>"
_BOV_TOKEN = "<|vid_start|>"
_EOV_TOKEN = "<|vid_end|>"
_VID_PAD_TOKEN = "<|video_pad|>"
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _resolve_special(tokenizer: Any, token: str, fallback_attr: str | None) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid == tokenizer.unk_token_id:
        if fallback_attr is not None:
            tid = getattr(tokenizer, fallback_attr, None)
    if tid is None:
        raise ValueError(f"Could not resolve special token {token!r}")
    return int(tid)


class TemporalInterleavedDataset(Dataset):
    """Map-style JSONL dataset for temporal mixed-modal prediction."""

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        tokenizer,
        image_size: int | tuple[int, int] = 512,
        max_text_length: int = 8192,
        max_images: int = 5,
        num_frames: int = 1,
        center_crop: bool = True,
        num_image_tokens: int | None = None,
        include_time_token: bool = True,
        noise_all_images_prob: float = 0.3,
        min_prefix_images: int = 1,
        max_segment_text_tokens: int = 256,
        train_image_boundary_tokens: bool = True,
        condition_dropout_prob: float = 0.0,
        drop_prefix_text_prob: float = 0.0,
        drop_prefix_visual_prob: float = 0.0,
        tuna_token_ids: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        )
        self.num_frames = int(num_frames)
        self.max_text_length = max_text_length
        self.max_images = max_images
        self.noise_all_images_prob = noise_all_images_prob
        self.min_prefix_images = min_prefix_images
        self.max_segment_text_tokens = max_segment_text_tokens
        self.train_image_boundary_tokens = train_image_boundary_tokens
        self.condition_dropout_prob = condition_dropout_prob
        self.drop_prefix_text_prob = drop_prefix_text_prob
        self.drop_prefix_visual_prob = drop_prefix_visual_prob
        self.records = self._load_jsonl(jsonl_path)
        self.image_transform = build_image_transform(self.image_size, center_crop)

        if num_image_tokens is None:
            patch = 16
            num_image_tokens = (
                self.num_frames
                * (self.image_size[0] // patch)
                * (self.image_size[1] // patch)
            )
            if include_time_token:
                num_image_tokens += 1
        self.num_image_tokens = num_image_tokens

        if tuna_token_ids is not None:
            self.bos_id = tuna_token_ids["bos_id"]
            self.eos_id = tuna_token_ids["eos_id"]
            self.pad_id = tokenizer.pad_token_id or self.eos_id
            self.boi_id = tuna_token_ids["boi_id"]
            self.eoi_id = tuna_token_ids["eoi_id"]
            self.bov_id = tuna_token_ids.get("bov_id", self.boi_id)
            self.eov_id = tuna_token_ids.get("eov_id", self.eoi_id)
            self.img_pad_id = tuna_token_ids["img_pad_id"]
            self.vid_pad_id = tuna_token_ids.get("vid_pad_id", self.img_pad_id)
        else:
            self.bos_id = _resolve_special(tokenizer, tokenizer.bos_token or "<|bos|>", "bos_token_id")
            self.eos_id = _resolve_special(tokenizer, tokenizer.eos_token or "<|eos|>", "eos_token_id")
            self.pad_id = _resolve_special(
                tokenizer,
                tokenizer.pad_token or tokenizer.eos_token or "<|pad|>",
                "pad_token_id",
            )
            self.boi_id = _resolve_special(tokenizer, _BOI_TOKEN, None)
            self.eoi_id = _resolve_special(tokenizer, _EOI_TOKEN, None)
            self.bov_id = _resolve_special(tokenizer, _BOV_TOKEN, None)
            self.eov_id = _resolve_special(tokenizer, _EOV_TOKEN, None)
            self.img_pad_id = _resolve_special(tokenizer, _IMG_PAD_TOKEN, None)
            self.vid_pad_id = _resolve_special(tokenizer, _VID_PAD_TOKEN, None)

        logger.info(
            "TemporalInterleavedDataset: loaded %d records from %s "
            "(image_size=%s, num_frames=%d, max_images=%d, num_image_tokens=%d)",
            len(self.records),
            jsonl_path,
            self.image_size,
            self.num_frames,
            max_images,
            self.num_image_tokens,
        )

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("%s:%d: skipping malformed JSON: %s", path, ln, e)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.image_root, rel_or_abs)

    def _load_image(self, rel_or_abs: str) -> Image.Image:
        path = self._resolve_image_path(rel_or_abs)
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB") if img.mode != "RGB" else img

    def _load_frames(self, rel_or_abs: str) -> list[Image.Image]:
        path = self._resolve_image_path(rel_or_abs)
        if os.path.isfile(path) and path.lower().endswith(_VIDEO_EXTENSIONS):
            return self._load_frames_from_video(path)
        if os.path.isdir(path):
            return self._load_frames_from_dir(path)
        return [self._load_image(path)]

    def _load_frames_from_video(self, video_file: str) -> list[Image.Image]:
        import cv2

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_file}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video has no frames: {video_file}")
        if total_frames >= self.num_frames:
            indices = [int(i * total_frames / self.num_frames) for i in range(self.num_frames)]
        else:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(total_frames - 1)

        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                raise RuntimeError(f"Failed to read frame {idx} from {video_file}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
        return frames

    def _load_frames_from_dir(self, frame_dir: str) -> list[Image.Image]:
        frame_files = sorted(
            f
            for f in os.listdir(frame_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not frame_files:
            raise RuntimeError(f"Frame directory has no images: {frame_dir}")
        if len(frame_files) > self.num_frames:
            indices = [int(i * len(frame_files) / self.num_frames) for i in range(self.num_frames)]
            frame_files = [frame_files[i] for i in indices]
        while len(frame_files) < self.num_frames:
            frame_files.append(frame_files[-1])
        frames = []
        for fname in frame_files:
            with Image.open(os.path.join(frame_dir, fname)) as img:
                img.load()
                frames.append(img.convert("RGB") if img.mode != "RGB" else img)
        return frames

    def _visual_to_tensor(self, item: dict[str, Any]) -> torch.Tensor:
        if item["type"] == "image":
            path = str(item["path"])
            frames = [self._load_image(path)]
        else:
            path = str(item.get("path") or item.get("video"))
            frames = self._load_frames(path)
        if len(frames) == 1 and self.num_frames > 1:
            frames = [frames[0].copy() for _ in range(self.num_frames)]
        elif len(frames) != self.num_frames:
            if len(frames) > self.num_frames:
                indices = [int(i * len(frames) / self.num_frames) for i in range(self.num_frames)]
                frames = [frames[i] for i in indices]
            while len(frames) < self.num_frames:
                frames.append(frames[-1].copy())
        frame_tensors = [self.image_transform(frame) for frame in frames]
        return torch.stack(frame_tensors, dim=1)

    @staticmethod
    def _is_visual(item: dict[str, Any]) -> bool:
        return item["type"] in {"image", "video"}

    def _normalise_sequence(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(record.get("sequence"), list):
            seq = []
            for item in record["sequence"]:
                kind = str(item.get("type", "")).lower()
                if kind == "text":
                    seq.append({"type": "text", "text": str(item.get("text", "")), **item})
                elif kind == "image":
                    path = item.get("path") or item.get("image")
                    if path:
                        seq.append({"type": "image", "path": path, **item})
                elif kind == "video":
                    path = item.get("path") or item.get("video")
                    if path:
                        seq.append({"type": "video", "path": path, **item})
                else:
                    raise ValueError(f"Unsupported sequence item type: {kind!r}")
            return seq

        images = record.get("images") or record.get("image_paths")
        captions = record.get("captions") or record.get("texts")
        if isinstance(images, list) and isinstance(captions, list):
            seq = []
            for caption, image in zip(captions, images):
                seq.append({"type": "text", "text": str(caption)})
                seq.append({"type": "image", "path": image})
            return seq
        videos = record.get("videos") or record.get("video_paths")
        if isinstance(videos, list) and isinstance(captions, list):
            seq = []
            for caption, video in zip(captions, videos):
                seq.append({"type": "text", "text": str(caption)})
                seq.append({"type": "video", "path": video})
            return seq
        raise ValueError("Temporal interleaved records need `sequence` or `images` + `captions`")

    def _crop_to_image_budget(self, seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
        image_positions = [i for i, item in enumerate(seq) if self._is_visual(item)]
        if len(image_positions) <= self.max_images:
            return seq
        start_image = random.randint(0, len(image_positions) - self.max_images)
        start = image_positions[start_image]
        if start > 0 and seq[start - 1]["type"] == "text":
            start -= 1
        end = image_positions[start_image + self.max_images - 1] + 1
        return seq[start:end]

    def _assign_targets(self, seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any("target" in item for item in seq):
            return [{**item, "target": bool(item.get("target", False))} for item in seq]

        image_seen = 0
        image_count = sum(1 for item in seq if self._is_visual(item))
        if image_count == 0:
            return [{**item, "target": True} for item in seq]

        if random.random() < self.noise_all_images_prob or image_count <= self.min_prefix_images:
            split_after_images = 0
        else:
            max_prefix = max(self.min_prefix_images, image_count - 1)
            split_after_images = random.randint(self.min_prefix_images, max_prefix)

        out = []
        for item in seq:
            is_target = image_seen >= split_after_images
            out.append({**item, "target": is_target})
            if self._is_visual(item):
                image_seen += 1
        return out

    def _apply_condition_dropout(self, seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.condition_dropout_prob <= 0 and self.drop_prefix_text_prob <= 0 and self.drop_prefix_visual_prob <= 0:
            return seq
        use_full_dropout = random.random() < self.condition_dropout_prob
        out = []
        for item in seq:
            if item.get("target", False):
                out.append(item)
                continue
            if item["type"] == "text" and (use_full_dropout or random.random() < self.drop_prefix_text_prob):
                out.append({**item, "text": "", "tokens": []})
            elif self._is_visual(item) and (
                (use_full_dropout and self.drop_prefix_visual_prob > 0)
                or random.random() < self.drop_prefix_visual_prob
            ):
                out.append({**item, "drop_visual": True})
            else:
                out.append(item)
        return out

    def _tokenize_segments(self, seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for item in seq:
            if item["type"] == "text":
                ids = self.tokenizer(
                    str(item.get("text", "")),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_segment_text_tokens,
                )["input_ids"]
                if ids:
                    out.append({**item, "tokens": list(ids)})
            else:
                out.append(item)
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        seq = self._normalise_sequence(record)
        seq = self._crop_to_image_budget(seq)
        seq = self._assign_targets(seq)
        seq = self._tokenize_segments(seq)
        seq = self._apply_condition_dropout(seq)

        images = []
        for item in seq:
            if self._is_visual(item):
                if item.get("drop_visual", False):
                    images.append(torch.zeros(3, self.num_frames, self.image_size[0], self.image_size[1]))
                else:
                    images.append(self._visual_to_tensor(item))
        if not images:
            raise ValueError("Temporal interleaved sample must contain at least one image")
        while len(images) < self.max_images:
            images.append(torch.zeros(3, self.num_frames, self.image_size[0], self.image_size[1]))

        tt, tl, mp, tm, im, image_target_masks = format_temporal_interleaved_sequence(
            segments=seq,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            boi_id=self.boi_id,
            eoi_id=self.eoi_id,
            pad_id=self.pad_id,
            img_pad_id=self.img_pad_id,
            num_image_tokens=self.num_image_tokens,
            max_seq_len=self.max_text_length,
            max_images=self.max_images,
            train_image_boundary_tokens=self.train_image_boundary_tokens,
            bov_id=self.bov_id,
            eov_id=self.eov_id,
            vid_pad_id=self.vid_pad_id,
        )

        return {
            "images": torch.stack(images[: self.max_images]),
            "text_tokens": tt.long(),
            "text_labels": tl.long(),
            "text_masks": tm.bool(),
            "image_masks": im.bool(),
            "image_target_masks": image_target_masks.bool(),
            "modality_positions": mp.long(),
            "data_type": "temporal_interleaved",
            "sentence": record.get("id", ""),
        }
