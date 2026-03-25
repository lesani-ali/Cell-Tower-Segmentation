import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..logging import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["background", "antenna"]
NUM_CLASSES = len(CLASS_NAMES)


def _load_mask(path: str) -> np.ndarray:
    """Load a mask image as a 2-D binary uint8 array (non-zero = foreground)."""
    img = cv2.imread(path)
    return (img.sum(axis=-1) > 0).astype(np.uint8)


def _fmt(v: float) -> str:
    return f"{v:.4f}" if not np.isnan(v) else "N/A"


class Eval:

    def __init__(self, gt_dir: str, output_report: Optional[str] = None) -> None:
        self.gt_dir = Path(gt_dir)
        self.output_report = output_report
        self.confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        self._processed = 0

        if not self.gt_dir.exists():
            logger.warning(f"GT directory not found: {self.gt_dir} — evaluation disabled.")

    def update(self, pred: np.ndarray, image_name: str) -> None:
        """
        Accumulate one predicted mask into the confusion matrix.

        Args:
            pred (np.ndarray): 2-D mask array (non-zero = antenna).
            image_name (str): Filename used to locate the matching GT mask.
        """
        if not self.gt_dir.exists():
            return

        gt_path = self._find_gt(Path(image_name).stem)
        if gt_path is None:
            logger.warning(f"No GT mask found for '{image_name}' in {self.gt_dir}")
            return

        gt = _load_mask(str(gt_path))

        if pred.shape != gt.shape:
            logger.warning(f"[{image_name}] shape mismatch — resizing prediction to {gt.shape}.")
            from PIL import Image
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), resample=0
                ),
                dtype=np.uint8,
            )

        pred = pred.flatten().astype(int)
        target = gt.flatten().astype(int)

        # Filter out invalid or "ignore" labels (e.g., 255)
        mask = (target >= 0) & (target < NUM_CLASSES)

        hist = np.bincount(
            NUM_CLASSES * target[mask] + pred[mask],
            minlength=NUM_CLASSES ** 2,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        self.confusion_matrix += hist
        self._processed += 1

    def compute(self) -> dict:
        """
        Compute global IoU and Dice from the accumulated confusion matrix.
        """
        tp = np.diag(self.confusion_matrix).astype(float)
        fp = self.confusion_matrix.sum(axis=0).astype(float) - tp
        fn = self.confusion_matrix.sum(axis=1).astype(float) - tp

        iou_per_class = tp / np.maximum(tp + fp + fn, 1e-7)
        dice_per_class = (2 * tp) / np.maximum(2 * tp + fp + fn, 1e-7)

        return {
            "iou_per_class": iou_per_class,
            "dice_per_class": dice_per_class,
            "miou": float(np.mean(iou_per_class)),
            "mdice": float(np.mean(dice_per_class)),
        }

    def reset(self) -> None:
        """Reset the confusion matrix."""
        self.confusion_matrix[:] = 0
        self._processed = 0

    def finalize(self) -> None:
        """
        Compute final metrics, print results, and save report.
        """
        if self._processed == 0:
            logger.warning("Eval: no images were evaluated.")
            return

        stats = self.compute()
        self._log(stats)

        if self.output_report:
            self._save(stats, self.output_report)

    def _log(self, stats: dict) -> None:
        sep = "=" * 40
        logger.info(f"\n{sep}")
        logger.info(f"  Evaluation Results  ({self._processed} images)")
        logger.info(sep)
        logger.info("  IoU:")
        for i, name in enumerate(CLASS_NAMES):
            logger.info(f"    {name:<14}: {_fmt(stats['iou_per_class'][i])}")
        logger.info("  Dice:")
        for i, name in enumerate(CLASS_NAMES):
            logger.info(f"    {name:<14}: {_fmt(stats['dice_per_class'][i])}")
        logger.info(f"  {'mIoU':<16}: {_fmt(stats['miou'])}")
        logger.info(f"  {'mDice':<16}: {_fmt(stats['mdice'])}")
        logger.info(sep)

    def _save(self, stats: dict, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        lines = [
            f"Evaluation Results  ({self._processed} images)",
            "=" * 40,
            "IoU:",
        ]
        for i, name in enumerate(CLASS_NAMES):
            lines.append(f"  {name:<14}: {_fmt(stats['iou_per_class'][i])}")
        lines.append("Dice:")
        for i, name in enumerate(CLASS_NAMES):
            lines.append(f"  {name:<14}: {_fmt(stats['dice_per_class'][i])}")
        lines.append(f"{'mIoU':<16}: {_fmt(stats['miou'])}")
        lines.append(f"{'mDice':<16}: {_fmt(stats['mdice'])}")

        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Evaluation report saved → {output_path}")

    def _find_gt(self, stem: str) -> Optional[Path]:
        for ext in ("png", "jpg", "tif", "jpeg", "PNG", "JPG"):
            p = self.gt_dir / f"{stem}.{ext}"
            if p.exists():
                return p
        return None
