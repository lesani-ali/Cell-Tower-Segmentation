import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..logging import get_logger

logger = get_logger(__name__)

# IoU thresholds for mAP@[0.5:0.95]
IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)   # [0.50, 0.55, …, 0.95]

def _poly_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    """Rasterize a COCO polygon segmentation to a binary (bool) mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def _pairwise_iou(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between M predicted masks and N GT masks.

    Strategy: loop over M predictions, broadcast against all N GT masks at
    once. This keeps peak memory to O(N * H * W) rather than O(M * N * H * W).

    Args:
        pred_masks (np.ndarray): (M, H, W) bool
        gt_masks (np.ndarray): (N, H, W) bool

    Returns:
        np.ndarray: (M, N) float32 IoU matrix
    """
    M, N = len(pred_masks), len(gt_masks)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)

    gt_areas = gt_masks.sum(axis=(1, 2)).astype(np.float32)   # (N,)
    iou_matrix = np.zeros((M, N), dtype=np.float32)

    for i in range(M):
        inter = np.logical_and(pred_masks[i], gt_masks).sum(axis=(1, 2)).astype(np.float32)  # (N,)
        union = float(pred_masks[i].sum()) + gt_areas - inter
        iou_matrix[i] = inter / np.maximum(union, 1e-7)

    return iou_matrix


def _greedy_match(
    pred_scores: np.ndarray,
    iou_matrix: np.ndarray,
    iou_thresh: float = 0.5,
) -> Tuple[np.ndarray, List[float]]:
    """
    Greedy matching: highest-confidence prediction first.

    Args:
        pred_scores (np.ndarray): (M,) confidence scores.
        iou_matrix (np.ndarray): (M, N) IoU values.
        iou_thresh (float): IoU threshold for a valid match.

    Returns:
        is_tp (np.ndarray): (M,) bool — which predictions are TPs.
        matched_ious (list[float]): IoU value of each matched pair.
    """
    M, N = iou_matrix.shape
    order = np.argsort(-pred_scores)          # descending score
    is_tp = np.zeros(M, dtype=bool)
    matched_gt = np.zeros(N, dtype=bool)
    matched_ious = []

    for orig_idx in order:
        row = iou_matrix[orig_idx].copy()
        row[matched_gt] = -1.0               # mask already-matched GT
        best_gt = int(np.argmax(row))

        if row[best_gt] >= iou_thresh:
            is_tp[orig_idx] = True
            matched_gt[best_gt] = True
            matched_ious.append(float(iou_matrix[orig_idx, best_gt]))

    return is_tp, matched_ious


def _compute_ap_101(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """
    Compute Average Precision using 101-point COCO interpolation.

    Args:
        scores (np.ndarray): (K,) prediction scores.
        is_tp (np.ndarray): (K,) bool TP flags.
        n_gt (int): total GT instances.

    Returns:
        float: AP value (NaN if no GT).
    """
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0

    order = np.argsort(-scores)
    tp_cum = np.cumsum(is_tp[order].astype(float))
    fp_cum = np.cumsum((~is_tp[order]).astype(float))

    precision = tp_cum / (tp_cum + fp_cum)  
    recall = tp_cum / n_gt

    # 101-point interpolation (COCO standard)
    ap = 0.0
    for thr in np.linspace(0.0, 1.0, 101):
        mask = recall >= thr
        ap += precision[mask].max() if mask.any() else 0.0
    return ap / 101.0


class Eval:
    """
    COCO-style instance segmentation evaluator.

    Metrics:
        AP@0.5, AP@0.75, mAP@[0.5:0.95],
        Precision@0.5, Recall@0.5, Mean matched mask IoU@0.5

    Args:
        gt_path (str): Path to the COCO-format annotation JSON.
        output_report (str | None): Optional path to save the text report.
    """

    def __init__(self, gt_path: str, output_report: Optional[str] = None) -> None:
        self.gt_path = Path(gt_path)
        self.output_report = output_report

        # Per-image accumulated data: (pred_scores (M,), iou_matrix (M,N), n_gt)
        self._data: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self._processed = 0

        # 2×2 confusion matrix for global semantic IoU / Dice
        # Rows = GT class, Cols = Pred class  (0=background, 1=antenna)
        self._conf: np.ndarray = np.zeros((2, 2), dtype=np.int64)

        # GT index: filename → {height, width, annotations[]}
        self._gt: Dict[str, dict] = {}
        self._load_gt()

    def _load_gt(self) -> None:
        if not self.gt_path.exists():
            logger.warning(f"GT JSON not found: {self.gt_path} — evaluation disabled.")
            return

        with open(self.gt_path) as f:
            data_json = json.load(f)

        ann_by_img: Dict[int, list] = {}
        for ann in data_json["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        for img_info in data_json["images"]:
            self._gt[img_info["file_name"]] = {
                "height": img_info["height"],
                "width": img_info["width"],
                "anns": ann_by_img.get(img_info["id"], []),
            }

        n_ann = sum(len(v["anns"]) for v in self._gt.values())
        logger.info(
            f"GT loaded: {len(self._gt)} images, {n_ann} instances from {self.gt_path.name}"
        )

    def _get_gt_masks(self, image_name: str) -> Optional[np.ndarray]:
        """
        Rasterize GT polygons for an image.

        Args:
            image_name (str): Filename (with extension) of the query image.

        Returns:
            np.ndarray | None: (N, H, W) bool, or None if GT not found.
        """
        info = self._gt.get(image_name)
        if info is None:
            # Fall back: match by stem
            stem = Path(image_name).stem
            info = next(
                (v for k, v in self._gt.items() if Path(k).stem == stem), None
            )
        if info is None:
            return None

        H, W, anns = info["height"], info["width"], info["anns"]
        if not anns:
            return np.zeros((0, H, W), dtype=bool)

        return np.stack(
            [_poly_to_mask(ann["segmentation"], H, W) for ann in anns], axis=0
        )

    def update(self, output: dict, image_name: str) -> None:
        """
        Accumulate predictions for one image.

        Args:
            output (dict): Keys "masks" (M, H, W) bool, "scores" (M,) float.
            image_name (str): Filename used to look up GT in the JSON.
        """
        if not self.gt_path.exists():
            return

        pred_masks = output.get("masks", np.empty((0, 0, 0), dtype=bool))
        pred_scores = np.asarray(output.get("scores", []), dtype=np.float32)

        gt_masks = self._get_gt_masks(image_name)
        if gt_masks is None:
            logger.warning(f"No GT found for '{image_name}' — skipping.")
            return

        n_gt = len(gt_masks)

        # Resize predictions to match GT resolution if needed
        if len(pred_masks) > 0 and n_gt > 0:
            ph, pw = pred_masks.shape[1], pred_masks.shape[2]
            gh, gw = gt_masks.shape[1], gt_masks.shape[2]
            if (ph, pw) != (gh, gw):
                logger.warning(
                    f"[{image_name}] pred size {(ph,pw)} ≠ GT size {(gh,gw)}, resizing."
                )
                pred_masks = np.stack([
                    cv2.resize(m.astype(np.uint8), (gw, gh),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    for m in pred_masks
                ], axis=0)

        # Compute pairwise IoU (M, N) — store only this small matrix
        iou_matrix = _pairwise_iou(pred_masks, gt_masks)
        self._data.append((pred_scores, iou_matrix, n_gt))

        # Combine all masks into binary images and update confusion matrix
        if n_gt > 0:
            H, W = gt_masks.shape[1], gt_masks.shape[2]
        else:
            H, W = pred_masks.shape[1], pred_masks.shape[2]

        pred_bin = (pred_masks.any(axis=0).astype(np.uint8)
                    if len(pred_masks) > 0
                    else np.zeros((H, W), dtype=np.uint8))
        gt_bin = (gt_masks.any(axis=0).astype(np.uint8)
                  if n_gt > 0
                  else np.zeros((H, W), dtype=np.uint8))

        hist = np.bincount(
            2 * gt_bin.flatten() + pred_bin.flatten(),
            minlength=4,
        ).reshape(2, 2)
        self._conf += hist

        self._processed += 1

    def compute(self) -> dict:
        """
        Compute all instance segmentation metrics from accumulated data.

        Returns:
            dict: ap50, ap75, map, precision50, recall50, mean_iou50.
        """
        total_gt = sum(n for _, _, n in self._data)

        ap_per_thresh: List[float] = []

        # Collect per-threshold data; also keep @0.5 for precision/recall/iou
        all_scores_50: List[float] = []
        all_is_tp_50:  List[bool]  = []
        matched_ious_50: List[float] = []

        for t_idx, iou_thresh in enumerate(IOU_THRESHOLDS):
            scores_all: List[float] = []
            is_tp_all:  List[bool]  = []
            n_gt_total = 0

            for pred_scores, iou_matrix, n_gt in self._data:
                n_gt_total += n_gt
                M = len(pred_scores)

                if M == 0:
                    continue

                if n_gt == 0:
                    # All predictions are FP
                    scores_all.extend(pred_scores.tolist())
                    is_tp_all.extend([False] * M)
                    continue

                is_tp, matched = _greedy_match(pred_scores, iou_matrix, iou_thresh)
                scores_all.extend(pred_scores.tolist())
                is_tp_all.extend(is_tp.tolist())

                if t_idx == 0:   # threshold == 0.5
                    matched_ious_50.extend(matched)

            if t_idx == 0:
                all_scores_50 = scores_all[:]
                all_is_tp_50  = is_tp_all[:]

            ap = _compute_ap_101(
                np.array(scores_all, dtype=np.float32),
                np.array(is_tp_all, dtype=bool),
                n_gt_total,
            )
            ap_per_thresh.append(ap)

        # mAP over valid thresholds
        valid = [v for v in ap_per_thresh if not np.isnan(v)]
        map_score = float(np.mean(valid)) if valid else float("nan")

        ap50 = ap_per_thresh[0]    # index 0 → thresh 0.50
        ap75 = ap_per_thresh[5]    # index 5 → thresh 0.75

        # Precision and Recall @0.5
        if all_scores_50:
            t = np.array(all_is_tp_50, dtype=bool)
            cum_tp = float(t.sum())
            cum_fp = float((~t).sum())
            precision50 = cum_tp / max(cum_tp + cum_fp, 1e-7)
            recall50 = cum_tp / max(total_gt, 1e-7)
        else:
            precision50 = recall50 = float("nan")

        mean_iou50 = float(np.mean(matched_ious_50)) if matched_ious_50 else float("nan")

        # Global semantic IoU and Dice from confusion matrix
        tp = np.diag(self._conf).astype(float)             # [TN, TP]
        fp = self._conf.sum(axis=0).astype(float) - tp     # FP per class
        fn = self._conf.sum(axis=1).astype(float) - tp     # FN per class

        iou = tp / np.maximum(tp + fp + fn, 1e-7)          # (2,)
        dice = (2 * tp) / np.maximum(2 * tp + fp + fn, 1e-7)  # (2,)

        return {
            # Instance metrics
            "ap50": float(ap50),
            "ap75": float(ap75),
            "map": map_score,
            "precision50": precision50,
            "recall50": recall50,
            "mean_iou50": mean_iou50,
            # Semantic metrics (combined binary masks)
            "iou_fg": float(iou[1]),
            "dice_fg": float(dice[1]),
        }

    def reset(self) -> None:
        """Reset all accumulators."""
        self._data.clear()
        self._conf[:] = 0
        self._processed = 0

    def finalize(self) -> None:
        """Compute final metrics, print results, and optionally save report."""
        if self._processed == 0:
            logger.warning("Eval: no images were evaluated.")
            return

        stats = self.compute()
        self._log(stats)
        if self.output_report:
            self._save(stats, self.output_report)

    def _report_lines(self, stats: dict) -> List[str]:
        w = 30
        return [
            f"  --- Instance Segmentation Metrics ---",
            f"  {'AP@0.5':<{w}}: {stats['ap50']:0.4f}",
            f"  {'AP@0.75':<{w}}: {stats['ap75']:0.4f}",
            f"  {'mAP@[0.5:0.95]':<{w}}: {stats['map']:0.4f}",
            f"  {'Precision@0.5':<{w}}: {stats['precision50']:0.4f}",
            f"  {'Recall@0.5':<{w}}: {stats['recall50']:0.4f}",
            f"  {'Mean Matched IoU@0.5':<{w}}: {stats['mean_iou50']:0.4f}",
            f"  --- Semantic Mask Quality (binary union) ---",
            f"  {'IoU  antenna':<{w}}: {stats['iou_fg']:0.4f}",
            f"  {'Dice antenna':<{w}}: {stats['dice_fg']:0.4f}",
        ]

    def _log(self, stats: dict) -> None:
        sep = "=" * 50
        logger.info(f"\n{sep}")
        logger.info(f"  Instance Segmentation Evaluation  ({self._processed} images)")
        logger.info(sep)
        for line in self._report_lines(stats):
            logger.info(line)
        logger.info(sep)

    def _save(self, stats: dict, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        header = [
            f"Instance Segmentation Evaluation  ({self._processed} images)",
            "=" * 50,
        ]
        with open(output_path, "w") as f:
            f.write("\n".join(header + self._report_lines(stats)) + "\n")
        logger.info(f"Evaluation report saved → {output_path}")
