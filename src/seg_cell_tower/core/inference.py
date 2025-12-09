from typing import Any
import numpy as np
from PIL import Image

from ..models.saliency import SaliencyDetectionModel
from ..models.depth import DepthModel
from ..models.object_detection import ObjectDetectionModel
from ..models.segmentation import SegmentationModel
from ..core.postprocessing import add_missed_info, post_process_boxes


def run_inference(
    image: Image.Image,
    saliency_model: SaliencyDetectionModel,
    depth_model: DepthModel,
    object_detection_model: ObjectDetectionModel,
    segmentation_model: SegmentationModel,
    config: Any
) -> np.ndarray:
    """
    Run inference on a single image and return masks.
    
    :param image: Input PIL Image
    :param saliency_model: Saliency detection model
    :param depth_model: Depth estimation model
    :param object_detection_model: Object detection model
    :param segmentation_model: Segmentation model
    :param config: Configuration object
    :return: Segmentation masks
    """
    image_height = image.height

    # Step 1: Saliency Detection
    saliency_img = saliency_model(image)

    # Step 2: Depth Estimation
    depth_map = depth_model(image)

    # Step 3: Add missed information
    no_background_img = add_missed_info(
        depth_map, saliency_img, image,
        config.recover_info_threshold
    )

    # Step 4: Object Detection
    boxes = object_detection_model(no_background_img)

    # Step 5: Post-process boxes
    boxes = post_process_boxes(
        boxes, image_height, depth_map,
        large_box_threshold=0.4,
        iou_threshold=0.5,
        farther_object_threshold=70
    )

    # Step 6: Segmentation
    masks = segmentation_model(image, boxes)

    return masks