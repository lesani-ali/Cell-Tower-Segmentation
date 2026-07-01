import os
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import Config
from .evaluation import SegmentationEvaluator
from .logging import get_logger
from .models import (
    DepthModel,
    ObjectDetectionModel,
    SaliencyDetectionModel,
    SegmentationModel,
)
from .postprocessing import add_missed_info, post_process_boxes
from .utils.io import load_image
from .utils.visualization import combine_image_with_mask, get_mask_img

logger = get_logger(__name__)


class SegmentationPipeline:
    def __init__(self, config: Config) -> None:

        logger.info("Loading saliency model…")
        self.saliency_model = SaliencyDetectionModel(config.models.saliency)

        logger.info("Loading depth model…")
        self.depth_model = DepthModel(config.models.depth)

        logger.info("Loading object-detection model…")
        self.object_detection_model = ObjectDetectionModel(
            config.models.object_detection
        )

        logger.info("Loading segmentation model (SAM)…")
        self.segmentation_model = SegmentationModel(config.models.segmentation)

        self.config = config

    def __call__(self, image: Image.Image) -> dict:
        return self.predict(image)

    def predict(self, image: Image.Image) -> dict:
        """
        Run the full pipeline on a single image.

        Parameters:
            image : PIL.Image.Image

        Returns:
            dict with keys:
                masks  : np.ndarray (N, H, W) bool — one per detected antenna.
                scores : np.ndarray (N,) float — detection confidence per mask.
        """
        image_height = image.height

        # Step 1: Saliency detection — remove background
        saliency_img = self.saliency_model(image)

        # Step 2: Depth estimation
        depth_map = self.depth_model(image)

        # Step 3: Recover missed foreground information using depth
        no_background_img = add_missed_info(
            depth_map,
            saliency_img,
            image,
            self.config.recover_info_threshold,
        )

        # Step 4: Detect antenna bounding boxes
        results = self.object_detection_model(no_background_img)

        # Step 5: Filter spurious / oversized / far-away boxes
        results = post_process_boxes(
            results,
            image_height,
            depth_map,
            large_box_threshold=0.4,
            iou_threshold=0.5,
            farther_object_threshold=70,
        )

        # Step 6: SAM segmentation prompt by box
        masks = self.segmentation_model(image, results["boxes"])

        return {"masks": masks, "scores": results["scores"]}

    def process_directory(
        self,
        input_img_dir: str,
        output_img_dir: str,
        output_mask_dir: str,
        gt_path: Optional[str] = None,
        output_report: Optional[str] = None,
    ) -> None:
        """
        Run inference on every image in input_img_dir.
        """
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        evaluator = None
        if gt_path:
            evaluator = SegmentationEvaluator(gt_path=gt_path, output_report=output_report)
            logger.info(f"Evaluation enabled — GT path: {gt_path}")

        input_imgs = sorted(os.listdir(input_img_dir))
        total_images = len(input_imgs)

        for idx, filename in enumerate(
            tqdm(input_imgs, desc="Processing images", ncols=100)
        ):
            if not filename.lower().endswith((".jpg", ".png")):
                continue

            logger.info(f"Processing image {idx + 1}/{total_images}: {filename}")

            img_path = os.path.join(input_img_dir, filename)
            base_name = os.path.splitext(filename)[0]
            in_img = load_image(img_path)

            # Inference
            output = self(in_img)

            # Build combined mask image (H×W, values 0 or 255)
            rgb_mask = get_mask_img(output["masks"], random_color=False)
            output_mask = Image.fromarray((rgb_mask * 255).astype(np.uint8))

            # Save mask
            mask_path = os.path.join(output_mask_dir, base_name + ".png")
            output_mask.save(mask_path)
            logger.info(f"Saved mask: {mask_path}")

            # Save overlay image
            overlay = combine_image_with_mask(in_img, output_mask)
            overlay_path = os.path.join(output_img_dir, base_name + ".png")
            overlay.save(overlay_path)
            logger.info(f"Saved overlay: {overlay_path}")

            # Evaluation — compare saved mask against GT
            if evaluator:
                evaluator.update(output, filename)

        # Print + save full evaluation report
        if evaluator is not None:
            evaluator.finalize()
