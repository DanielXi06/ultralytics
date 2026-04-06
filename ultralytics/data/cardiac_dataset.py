# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset


class CardiacDetectionDataset(YOLODataset):
    """YOLO detection dataset extended with per-image cardiac semantic masks."""

    def __init__(self, *args, data: dict | None = None, mode: str = "train", **kwargs):
        """Initialize dataset and resolve semantic mask files for each image."""
        self.mode = mode
        super().__init__(*args, data=data, task="detect", **kwargs)
        self.sem_files = [self._sem_path_from_image(Path(p)) for p in self.im_files]

    def _sem_root(self) -> Path | None:
        """Resolve semantic mask root directory from dataset configuration."""
        split = "train" if self.mode == "train" else "val" if self.mode == "val" else "test"
        data = self.data or {}
        key = f"sem_{split}"
        if data.get(key):
            sem_root = Path(data[key])
            if data.get("path") and not sem_root.is_absolute():
                sem_root = Path(data["path"]) / sem_root
            return sem_root
        if data.get("sem_masks"):
            sem_root = Path(data["sem_masks"])
            if data.get("path") and not sem_root.is_absolute():
                sem_root = Path(data["path"]) / sem_root
            return sem_root / split
        return None

    def _sem_path_from_image(self, image_path: Path) -> Path | None:
        """Resolve semantic mask path for one image."""
        sem_root = self._sem_root()
        if sem_root is not None:
            return sem_root / f"{image_path.stem}.png"

        # Fallback convention: .../images/<split>/xxx.jpg -> .../sem_masks/<split>/xxx.png
        parts = image_path.parts
        if "images" in parts:
            i = parts.index("images")
            sem_parts = list(parts)
            sem_parts[i] = "sem_masks"
            sem_path = Path(*sem_parts).with_suffix(".png")
            return sem_path
        return None

    def _load_semantic(self, i: int, hw: tuple[int, int]) -> np.ndarray:
        """Load one semantic mask and align it with the pre-transform image size."""
        h, w = hw
        sem = np.zeros((h, w), dtype=np.uint8)

        sem_path = self.sem_files[i]
        if sem_path and sem_path.exists():
            sem_loaded = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
            if sem_loaded is not None:
                if sem_loaded.shape != (h, w):
                    sem_loaded = cv2.resize(sem_loaded, (w, h), interpolation=cv2.INTER_NEAREST)
                sem = sem_loaded.astype(np.uint8)
        return sem

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Return detection labels plus a semantic mask before any spatial transforms."""
        label = super().get_image_and_label(index)
        label["semantic"] = self._load_semantic(index, label["img"].shape[:2])
        return label
