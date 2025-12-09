import numpy as np
from PIL import Image


def add_missed_info(
    depth_map: np.ndarray,
    saliency_img: Image.Image,
    image: Image.Image,
    threshold: int = 140
) -> np.ndarray:
    """
    Add missed information from the original image to the saliency image based on the depth map.

    :param depth_map: Depth map of the image.
    :param saliency_img: Saliency image.
    :param image: Original image.
    :param threshold: Depth threshold to determine recoverable information.
    :return: Updated saliency image with recovered information.
    """
    saliency_img = np.array(saliency_img)
    image = np.asarray(image)

    mask = depth_map > threshold

    saliency_img[mask] = image[mask]

    return saliency_img


def remove_farther_objects(
    depth_map: np.ndarray,
    boxes: np.ndarray,
    threshold: int
) -> np.ndarray:
    """
    Remove objects that are farther away based on the depth map.

    :param depth_map: Depth map of the image.
    :param image: Image to remove objects from.
    :param threshold: Depth threshold to determine objects to remove.
    :return: Image with farther objects removed.
    """
    keep_row = np.ones(len(boxes), dtype=bool)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Crop region
        roi = depth_map[y1:y2, x1:x2]

        if np.mean(roi) < threshold:
            keep_row[idx] = False

    return boxes[keep_row]


def post_process_boxes(
    boxes: np.ndarray,
    image_height: int,
    depth_map: np.ndarray,
    large_box_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    farther_object_threshold: int = 80
) -> np.ndarray:
    """filter_containing_boxes
    Post-process bounding boxes by removing large boxes and filtering containing boxes.

    :param boxes: Array of bounding boxes.
    :param large_box_threshold: Threshold for determining large boxes.
    :param iou_threshold: IoU threshold for filtering containing boxes.
    :return: Filtered array of bounding boxes.
    """
    boxes = remove_large_boxes(
        boxes, image_height, threshold=large_box_threshold
    )
    boxes = filter_nested_boxes(boxes, iou_threshold=iou_threshold)

    boxes = remove_farther_objects(
        depth_map, boxes, farther_object_threshold
    )
    return boxes


def compute_iou(boxes: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) for all pairs of boxes in a numpy array.

    :param boxes: Array of bounding boxes.
    :return: IoU matrix for all pairs of boxes.
    """
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])

    # Calculate intersection areas
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate the area of each box
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Union = Area1 + Area2 - Intersection
    # union = area[:, None] + area[None, :] - intersection

    # IoU = Intersection / Union
    # iou = intersection / np.maximum(union, 1e-6)  # Avoid division by zero

    iou = intersection / np.maximum(area[:, None], 1e-6)  # Avoid division by zero
    return iou


def remove_large_boxes(
    boxes: np.ndarray, img_height: int, threshold: float
) -> np.ndarray:
    """
    Remove large boxes based on width, height, and IoU.

    :param boxes: Array of bounding boxes.
    :param img_height: Height of the image.
    :param threshold: Threshold for determining large boxes.
    :return: Filtered array of bounding boxes.
    """

    largest_box_idx = np.argmax(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    )

    # Remove boxes that are too wide
    box_width = np.abs(boxes[:, 2] - boxes[:, 0])
    mask1 = box_width > threshold * np.abs(boxes[largest_box_idx, 2] - boxes[largest_box_idx, 0])

    # Remove boxes that are too tall
    box_height = np.abs(boxes[:, 3] - boxes[:, 1])
    mask2 = box_height > (img_height * threshold)

    # Remove boxes out of frame of the largest box
    iou = compute_iou(boxes)
    mask3 = iou[largest_box_idx] < 1e-5

    # Combine all masks
    mask = mask1 | mask2 | mask3

    return boxes[~mask]


def filter_nested_boxes(
    boxes: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """
    Filter out boxes that are inside other boxes with high IoU.

    :param boxes: Array of bounding boxes.
    :param iou_threshold: IoU threshold for filtering containing boxes.
    :return: Filtered array of bounding boxes.
    """
    keep_row = np.ones(len(boxes), dtype=bool)
    boxes = boxes[np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
    iou = compute_iou(boxes)

    for i in range(len(boxes)):
        # if not keep_row[i]:
        #     continue
        for j in range(i + 1, len(boxes)):
            if iou[j, i] > iou_threshold:
                keep_row[i] = False  # Mark the larger box as redundant
                break

    # print(boxes[keep_row])
    return boxes[keep_row]
