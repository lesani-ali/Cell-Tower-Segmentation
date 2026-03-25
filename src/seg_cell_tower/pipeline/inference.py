from typing import Any

import numpy as np
from PIL import Image

from ..models.saliency import SaliencyDetectionModel
from ..models.depth import DepthModel
from ..models.object_detection import ObjectDetectionModel
from ..models.segmentation import SegmentationModel
from ..postprocessing.box_ops import add_missed_info, post_process_boxes
from ..logging import get_logger

logger = get_logger(__name__)


def run_inference(
    image: Image.Image,
    saliency_model: SaliencyDetectionModel,
    depth_model: DepthModel,
    object_detection_model: ObjectDetectionModel,
    segmentation_model: SegmentationModel,
    config: Any,
) -> np.ndarray:
    """
    Run the full inference pipeline on a single image.
    """
    image_height = image.height

    # Step 1: Saliency detection — remove background
    saliency_img = saliency_model(image)

    # Step 2: Depth estimation
    depth_map = depth_model(image)

    # Step 3: Recover missed foreground information using depth
    no_background_img = add_missed_info(
        depth_map, saliency_img, image,
        config.recover_info_threshold,
    )

    # Step 4: Detect antenna bounding boxes
    boxes = object_detection_model(no_background_img)

    # Step 5: Filter spurious / oversized / far-away boxes
    boxes = post_process_boxes(
        boxes,
        image_height,
        depth_map,
        large_box_threshold=0.4,
        iou_threshold=0.5,
        farther_object_threshold=70,
    )

    # Step 6: SAM segmentation prompt by box
    masks = segmentation_model(image, boxes)

    return masks
