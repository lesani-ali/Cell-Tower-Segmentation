import os
from pathlib import Path
from typing import List, Optional

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
    return f"{v:.4f}" if not np.isnan(v) else "  N/A"


def _fmt_meanstd(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


class Eval:

    def __init__(self, gt_dir: str, output_report: Optional[str] = None) -> None:
        self.gt_dir = Path(gt_dir)
        self.output_report = output_report

        # Global confusion matrix (accumulated across all images)
        self.confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

        # Per-image IoU and Dice lists (one entry per class per image)
        self._per_image_iou: List[np.ndarray] = []   # each: shape (NUM_CLASSES,)
        self._per_image_dice: List[np.ndarray] = []

        self._processed = 0

        if not self.gt_dir.exists():
            logger.warning(f"GT directory not found: {self.gt_dir} — evaluation disabled.")

    def update(self, pred: np.ndarray, image_name: str) -> None:
        """
        Accumulate one predicted mask into the confusion matrix and per-image lists.

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

        pred_flat = pred.flatten().astype(int)
        target_flat = gt.flatten().astype(int)

        # Filter out invalid / ignore labels (e.g., 255)
        mask = (target_flat >= 0) & (target_flat < NUM_CLASSES)

        hist = np.bincount(
            NUM_CLASSES * target_flat[mask] + pred_flat[mask],
            minlength=NUM_CLASSES ** 2,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        # Accumulate global confusion matrix
        self.confusion_matrix += hist

        # Compute and store per-image IoU & Dice from this image's histogram
        tp = np.diag(hist).astype(float)
        fp = hist.sum(axis=0).astype(float) - tp
        fn = hist.sum(axis=1).astype(float) - tp

        iou_img = tp / np.maximum(tp + fp + fn, 1e-7)
        dice_img = (2 * tp) / np.maximum(2 * tp + fp + fn, 1e-7)

        self._per_image_iou.append(iou_img)
        self._per_image_dice.append(dice_img)
        self._processed += 1

    def compute(self) -> dict:
        """
        Compute all metrics from the accumulated confusion matrix and per-image lists.
        """
        tp = np.diag(self.confusion_matrix).astype(float)
        fp = self.confusion_matrix.sum(axis=0).astype(float) - tp
        fn = self.confusion_matrix.sum(axis=1).astype(float) - tp

        # Global metrics (from accumulated totals)
        iou_per_class = tp / np.maximum(tp + fp + fn, 1e-7)
        dice_per_class = (2 * tp) / np.maximum(2 * tp + fp + fn, 1e-7)
        precision_per_class = tp / np.maximum(tp + fp, 1e-7)
        recall_per_class = tp / np.maximum(tp + fn, 1e-7)

        # Per-image statistics (mean ± std across images, per class)
        img_iou = np.stack(self._per_image_iou, axis=0)   # (N, NUM_CLASSES)
        img_dice = np.stack(self._per_image_dice, axis=0)   # (N, NUM_CLASSES)

        iou_mean_per_class = img_iou.mean(axis=0)
        iou_std_per_class = img_iou.std(axis=0)
        dice_mean_per_class = img_dice.mean(axis=0)
        dice_std_per_class = img_dice.std(axis=0)

        return {
            # Global
            "iou_per_class": iou_per_class,
            "dice_per_class": dice_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "miou": float(np.mean(iou_per_class)),
            "mdice": float(np.mean(dice_per_class)),
            "mprecision": float(np.mean(precision_per_class)),
            "mrecall": float(np.mean(recall_per_class)),
            # Per-image mean ± std
            "iou_mean_per_class": iou_mean_per_class,
            "iou_std_per_class": iou_std_per_class,
            "dice_mean_per_class": dice_mean_per_class,
            "dice_std_per_class": dice_std_per_class,
        }

    def reset(self) -> None:
        """Reset all accumulators."""
        self.confusion_matrix[:] = 0
        self._per_image_iou.clear()
        self._per_image_dice.clear()
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
        sep = "=" * 56
        logger.info(f"\n{sep}")
        logger.info(f"  Evaluation Results  ({self._processed} images)")
        logger.info(sep)

        logger.info("  Global metrics (computed from dataset-level pixel counts):")
        logger.info(f"  {'Metric':<12}  {'background':>12}  {'antenna':>12}  {'mean':>10}")
        logger.info(f"  {'-'*50}")
        for metric, key in [("IoU", "iou"), ("Dice", "dice"), ("Precision", "precision"), ("Recall", "recall")]:
            per = stats[f"{key}_per_class"]
            mean = stats[f"m{key}"]
            row = "  ".join(_fmt(per[i]) for i in range(NUM_CLASSES))
            logger.info(f"  {metric:<12}  {_fmt(per[0]):>12}  {_fmt(per[1]):>12}  {_fmt(mean):>10}")

        logger.info(f"  Per-image statistics (mean ± std across {self._processed} images):")
        logger.info(f"  {'Metric':<12}  {'background':>20}  {'antenna':>20}")
        logger.info(f"  {'-'*56}")
        for metric, m_key, s_key in [
            ("IoU",  "iou_mean_per_class",  "iou_std_per_class"),
            ("Dice", "dice_mean_per_class", "dice_std_per_class"),
        ]:
            means = stats[m_key]
            stds  = stats[s_key]
            logger.info(
                f"  {metric:<12}  "
                f"{_fmt_meanstd(means[0], stds[0]):>20}  "
                f"{_fmt_meanstd(means[1], stds[1]):>20}"
            )
        logger.info(sep)

    def _save(self, stats: dict, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        lines = [
            f"Evaluation Results  ({self._processed} images)",
            "=" * 56,
            "",
            "Global metrics (from dataset-level pixel counts):",
            f"  {'Metric':<12}  {'background':>12}  {'antenna':>12}  {'mean':>10}",
            f"  {'-'*50}",
        ]
        for metric, key in [("IoU", "iou"), ("Dice", "dice"), ("Precision", "precision"), ("Recall", "recall")]:
            per  = stats[f"{key}_per_class"]
            mean = stats[f"m{key}"]
            lines.append(f"  {metric:<12}  {_fmt(per[0]):>12}  {_fmt(per[1]):>12}  {_fmt(mean):>10}")

        lines += [
            "",
            f"Per-image statistics (mean ± std across {self._processed} images):",
            f"  {'Metric':<12}  {'background':>20}  {'antenna':>20}",
            f"  {'-'*56}",
        ]
        for metric, m_key, s_key in [
            ("IoU",  "iou_mean_per_class",  "iou_std_per_class"),
            ("Dice", "dice_mean_per_class", "dice_std_per_class"),
        ]:
            means = stats[m_key]
            stds  = stats[s_key]
            lines.append(
                f"  {metric:<12}  "
                f"{_fmt_meanstd(means[0], stds[0]):>20}  "
                f"{_fmt_meanstd(means[1], stds[1]):>20}"
            )

        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Evaluation report saved → {output_path}")

    def _find_gt(self, stem: str) -> Optional[Path]:
        for ext in ("png", "jpg", "tif", "jpeg", "PNG", "JPG"):
            p = self.gt_dir / f"{stem}.{ext}"
            if p.exists():
                return p
        return None
