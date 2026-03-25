import os
from typing import Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..models import DepthModel, ObjectDetectionModel, SaliencyDetectionModel, SegmentationModel
from ..utils.io import load_image
from ..utils.visualization import combine_image_with_mask, get_mask_img
from ..logging import get_logger
from .inference import run_inference

logger = get_logger(__name__)


class SegmentationPipeline:

    def __init__(self, config: Any) -> None:

        logger.info("Loading saliency model…")
        self.saliency_model = SaliencyDetectionModel(config.models.saliency)

        logger.info("Loading depth model…")
        self.depth_model = DepthModel(config.models.depth)

        logger.info("Loading object-detection model…")
        self.object_detection_model = ObjectDetectionModel(config.models.object_detection)

        logger.info("Loading segmentation model (SAM)…")
        self.segmentation_model = SegmentationModel(config.models.segmentation)

        self.config = config

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.predict(image)

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Run the full pipeline on a single image.

        Parameters:
            image : PIL.Image.Image

        Returns:
            np.ndarray  shape (N, H, W) — boolean masks, one per detected antenna.
        """
        return run_inference(
            image,
            self.saliency_model,
            self.depth_model,
            self.object_detection_model,
            self.segmentation_model,
            self.config,
        )

    def process_directory(
        self,
        input_img_dir: str,
        output_img_dir: str,
        output_mask_dir: str,
        gt_dir: Optional[str] = None,
        output_report: Optional[str] = None,
    ) -> None:
        """
        Run inference on every image in input_img_dir.
        """
        os.makedirs(output_img_dir,  exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        evaluator = None
        if gt_dir:
            from ..evaluation.evaluator import Eval
            evaluator = Eval(gt_dir=gt_dir, output_report=output_report)
            logger.info(f"Evaluation enabled — GT dir: {gt_dir}")

        input_imgs = sorted(os.listdir(input_img_dir))
        total_images = len(input_imgs)

        for idx, filename in enumerate(tqdm(input_imgs, desc="Processing images", ncols=100)):
            if not filename.lower().endswith((".jpg", ".png")):
                continue

            logger.info(f"Processing image {idx + 1}/{total_images}: {filename}")

            img_path = os.path.join(input_img_dir, filename)
            base_name = os.path.splitext(filename)[0]
            in_img = load_image(img_path)

            # Inference
            masks = self(in_img)

            # Build combined mask image (H×W, values 0 or 255)
            rgb_mask, binary_mask = get_mask_img(masks)
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
                evaluator.update(binary_mask, filename)

        # Print + save full evaluation report
        if evaluator is not None:
            evaluator.finalize()

