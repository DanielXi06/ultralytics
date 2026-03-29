# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

from ultralytics.data import build_dataloader
from ultralytics.data.cardiac_dataset import CardiacDetectionDataset
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class CardiacDetectionTrainer(DetectionTrainer):
    """Detection trainer for cardiac lesion detection with auxiliary semantic masks."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize trainer with conservative defaults to reduce image-mask geometric mismatch risk."""
        overrides = {} if overrides is None else overrides
        defaults = {
            "task": "detect",
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
        }
        for k, v in defaults.items():
            overrides.setdefault(k, v)
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build cardiac detection dataset with semantic masks."""
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return CardiacDetectionDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            mode=mode,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not all((dataset.batch_shapes == dataset.batch_shapes[0]).ravel()):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def get_validator(self):
        """Return DetectionValidator with 4-component loss names including semantic loss."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "sem_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
