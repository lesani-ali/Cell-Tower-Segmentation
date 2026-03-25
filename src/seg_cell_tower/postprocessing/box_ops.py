import numpy as np
from PIL import Image


def add_missed_info(
    depth_map: np.ndarray,
    saliency_img: Image.Image,
    image: Image.Image,
    threshold: int = 140,
) -> np.ndarray:
    """
    Recover foreground pixels that saliency missed, guided by the depth map.
    """
    saliency_arr = np.array(saliency_img)
    image_arr = np.asarray(image)

    mask = depth_map > threshold
    saliency_arr[mask] = image_arr[mask]

    return saliency_arr


def bbox_iou(boxes: np.ndarray) -> np.ndarray:
    """
    Compute the containment-ratio IoU matrix for all pairs of boxes.

    For each pair (i, j) the value is:
        intersection(i, j) / area(i)

    This is intentionally asymmetric — it measures how much box *i* is
    contained within box *j*, used for nested-box filtering.

    Parameters
    ----------
    boxes : np.ndarray  Shape (N, 4) in xyxy format.

    Returns
    -------
    np.ndarray  Shape (N, N) IoU matrix.
    """
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return intersection / np.maximum(area[:, None], 1e-6)


def remove_large_boxes(
    boxes: np.ndarray,
    img_height: int,
    threshold: float,
) -> np.ndarray:
    """
    Remove boxes that are disproportionately large relative to the scene.
    """
    largest_idx = np.argmax(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    )

    box_width = np.abs(boxes[:, 2] - boxes[:, 0])
    box_height = np.abs(boxes[:, 3] - boxes[:, 1])

    mask_wide = box_width > threshold * np.abs(boxes[largest_idx, 2] - boxes[largest_idx, 0])
    mask_tall = box_height > img_height * threshold
    mask_nooverlap = bbox_iou(boxes)[largest_idx] < 1e-5

    return boxes[~(mask_wide | mask_tall | mask_nooverlap)]


def remove_farther_objects(
    depth_map: np.ndarray,
    boxes: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """
    Drop boxes whose ROI mean depth is below *threshold* (too far away).
    """
    keep = np.ones(len(boxes), dtype=bool)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = depth_map[y1:y2, x1:x2]
        if np.mean(roi) < threshold:
            keep[idx] = False
    return boxes[keep]


def filter_nested_boxes(
    boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    Remove boxes that are largely contained within a larger sibling box.
    """
    # Sort largest-first so outer boxes are processed first
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    boxes = boxes[order]
    iou = bbox_iou(boxes)
    keep = np.ones(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou[j, i] > iou_threshold:
                keep[i] = False
                break

    return boxes[keep]


def post_process_boxes(
    boxes: np.ndarray,
    image_height: int,
    depth_map: np.ndarray,
    large_box_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    farther_object_threshold: int = 80,
) -> np.ndarray:
    """
    Full post-processing chain for detected bounding boxes.
    """
    boxes = remove_large_boxes(boxes, image_height, threshold=large_box_threshold)
    boxes = filter_nested_boxes(boxes, iou_threshold=iou_threshold)
    boxes = remove_farther_objects(depth_map, boxes, farther_object_threshold)
    return boxes
