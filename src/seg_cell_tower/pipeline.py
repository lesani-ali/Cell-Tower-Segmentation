import os
from typing import Optional, Tuple

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
from .postprocessing import (
    build_tower_prior,
    get_roi_box,
    offset_boxes,
    post_process_boxes,
)
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

    def _detect_candidates(
        self,
        image: Image.Image,
        crop_box: Tuple[int, int, int, int],
    ) -> dict:
        image_arr = np.asarray(image)
        x1, y1, x2, y2 = crop_box
        crop = image_arr[y1:y2, x1:x2]

        if crop.size == 0:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "prompts": [],
            }

        detections = self.object_detection_model(crop)
        if len(detections["boxes"]) == 0:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "prompts": [],
            }

        return {
            "boxes": offset_boxes(detections["boxes"], crop_box).astype(np.float32),
            "scores": detections["scores"].astype(np.float32),
            "prompts": detections.get("prompts", []),
        }

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
        image_width = image.width

        # Step 1: Saliency and depth produce a coarse tower prior.
        saliency_img = self.saliency_model(image)
        depth_map = self.depth_model(image)
        tower_prior, saliency_mask = build_tower_prior(
            depth_map=depth_map,
            saliency_img=saliency_img,
            recover_threshold=self.config.recover_info_threshold,
        )

        # Step 2: Detect on original-image tower crops to keep the image natural.
        tower_roi = get_roi_box(tower_prior)
        results = self._detect_candidates(image, tower_roi)

        # Step 3: Re-score detections with tower-aware priors.
        results = post_process_boxes(
            results,
            image_shape=(image_height, image_width),
            depth_map=depth_map,
            nms_threshold=self.config.models.object_detection.nms_threshold,
            ignore_threshold=self.config.ignore_info_threshold,
        )

        # Step 4: Segment each surviving box with SAM.
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
            if len(output["masks"]) == 0:
                mask = Image.fromarray(
                    np.zeros((in_img.height, in_img.width, 4), dtype=np.uint8)
                )
                output_mask = mask.convert("RGB")
            else:
                rgb_mask = get_mask_img(output["masks"], random_color=False)
                mask = Image.fromarray((rgb_mask * 255).astype(np.uint8))
                output_mask = mask.convert("RGB")
    
            # Save mask
            mask_path = os.path.join(output_mask_dir, base_name + ".png")
            output_mask.save(mask_path)
            logger.info(f"Saved mask: {mask_path}")

            # Save overlay image
            overlay = combine_image_with_mask(in_img, mask)
            overlay_path = os.path.join(output_img_dir, base_name + ".png")
            overlay.save(overlay_path)
            logger.info(f"Saved overlay: {overlay_path}")

            # Evaluation — compare saved mask against GT
            if evaluator:
                evaluator.update(output, filename)

        # Print + save full evaluation report
        if evaluator is not None:
            evaluator.finalize()
