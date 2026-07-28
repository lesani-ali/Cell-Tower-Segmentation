from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def saliency_to_mask(
    saliency_img: Image.Image,
    white_threshold: int = 245,
) -> np.ndarray:
    """
    Convert the saliency output image into a foreground mask.
    """
    saliency_arr = np.asarray(saliency_img)
    return np.any(saliency_arr < white_threshold, axis=-1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 1:
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest_idx


def build_tower_prior(
    depth_map: np.ndarray,
    saliency_img: Image.Image,
    recover_threshold: int = 140,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a coarse tower prior from saliency and relative depth.

    Returns:
        tower_prior : (H, W) bool --> approximate tower region
        saliency_mask : (H, W) bool
    """
    saliency_mask = saliency_to_mask(saliency_img)
    height, width = depth_map.shape

    if np.any(saliency_mask):
        xs = np.where(saliency_mask.any(axis=0))[0]
        band_margin = max(8, int(0.08 * width))
        band_x1 = max(0, int(xs[0]) - band_margin)
        band_x2 = min(width, int(xs[-1]) + band_margin + 1)
        band_mask = np.zeros_like(saliency_mask)
        band_mask[:, band_x1:band_x2] = True
        recover_cutoff = max(
            recover_threshold,
            int(np.percentile(depth_map[saliency_mask], 70)),
        )
    else:
        band_mask = np.ones_like(saliency_mask, dtype=bool)
        recover_cutoff = max(recover_threshold, int(np.percentile(depth_map, 80)))

    recover_mask = depth_map >= recover_cutoff
    tower_prior = saliency_mask | (recover_mask & band_mask)

    if not np.any(tower_prior):
        tower_prior = saliency_mask | band_mask

    tower_prior = cv2.morphologyEx(
        tower_prior.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
    ).astype(bool)
    tower_prior = cv2.morphologyEx(
        tower_prior.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    ).astype(bool)

    largest = _largest_component(tower_prior)
    if np.any(largest):
        tower_prior = largest

    return tower_prior, saliency_mask


def get_roi_box(mask: np.ndarray, margin_ratio: float = 0.06) -> Tuple[int, int, int, int]:
    """
    Convert a binary mask into an expanded xyxy box.
    """
    height, width = mask.shape
    if not np.any(mask):
        return 0, 0, width, height

    ys, xs = np.where(mask)
    margin_x = max(8, int(width * margin_ratio))
    margin_y = max(8, int(height * margin_ratio))

    x1 = max(0, int(xs.min()) - margin_x)
    y1 = max(0, int(ys.min()) - margin_y)
    x2 = min(width, int(xs.max()) + margin_x + 1)
    y2 = min(height, int(ys.max()) + margin_y + 1)

    return x1, y1, x2, y2


def offset_boxes(boxes: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Move crop-local boxes back into image coordinates.
    """
    if len(boxes) == 0:
        return boxes

    x1, y1, _, _ = crop_box
    shifted = boxes.copy()
    shifted[:, [0, 2]] += x1
    shifted[:, [1, 3]] += y1
    return shifted


def bbox_iou(boxes: np.ndarray) -> np.ndarray:
    """
    Compute the containment-ratio IoU matrix for all pairs of boxes."""
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return intersection / np.maximum(area[:, None], 1e-6)


def remove_large_boxes(
    results: dict,
    img_height: int,
    threshold: float,
) -> dict:
    """
    Remove boxes that are disproportionately large relative to the scene.
    """
    boxes = results["boxes"]
    scores = results["scores"]
    prompts = results.get("prompts", [])
    
    largest_idx = np.argmax(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    )

    box_width = np.abs(boxes[:, 2] - boxes[:, 0])
    box_height = np.abs(boxes[:, 3] - boxes[:, 1])

    mask_wide = box_width > threshold * np.abs(boxes[largest_idx, 2] - boxes[largest_idx, 0])
    mask_tall = box_height > img_height * threshold
    mask_nooverlap = bbox_iou(boxes)[largest_idx] < 1e-5

    keep = ~(mask_wide | mask_tall | mask_nooverlap)
    results["boxes"] = boxes[keep].astype(np.float32)
    results["scores"] = scores[keep]
    results["prompts"] = [prompt for prompt, keep in zip(prompts, keep) if keep]

    return results


def filter_nested_boxes(
    results: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Remove boxes that are largely contained within a larger sibling box.
    """
    # Sort largest-first so outer boxes are processed first
    boxes = results["boxes"]
    scores = results["scores"]

    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    boxes = boxes[order]
    iou = bbox_iou(boxes)
    keep = np.ones(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou[j, i] > iou_threshold:
                keep[i] = False
                break

    results["boxes"] = boxes[keep]
    results["scores"] = scores[keep]
    return results


def remove_farther_objects(
    depth_map: np.ndarray,
    results: dict,
    threshold: int,
) -> dict:
    """
    Drop boxes whose ROI mean depth is below *threshold* (too far away).
    """
    boxes = results["boxes"]
    scores = results["scores"]

    keep = np.ones(len(boxes), dtype=bool)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = depth_map[y1:y2, x1:x2]
        if np.mean(roi) < threshold:
            keep[idx] = False

    results["boxes"] = boxes[keep]
    results["scores"] = scores[keep]
    return results


def post_process_boxes(
    results: dict,
    image_shape: Tuple[int, int],
    depth_map: np.ndarray,
    nms_threshold: float = 0.5,
    ignore_threshold: int = 80,
) -> dict:
    """
    Re-score detections using tower-aware priors, then apply NMS.
    """

    results = remove_large_boxes(
        results,
        img_height=image_shape[0],
        threshold=0.4,
    )

    results = filter_nested_boxes(results, iou_threshold=nms_threshold)

    results = remove_farther_objects(depth_map, results, ignore_threshold)

    return results