from typing import Any
from PIL import Image
import numpy as np
import os
import logging
from tqdm import tqdm

from ..models.saliency import SaliencyDetectionModel
from ..models.depth import DepthModel
from ..models.object_detection import ObjectDetectionModel
from ..models.segmentation import SegmentationModel
from ..core.inference import run_inference
from ..utils.io import load_image
from ..utils.visualization import (
    combine_image_with_mask,
    get_mask_img
)


class SegmentationPipeline(object):
    """Main pipeline for cell tower segmentation."""
    
    def __init__(self, config: Any):
        """Initialize pipeline with models."""
        self.saliency_model = SaliencyDetectionModel(
            config.models.saliency
        )
        self.depth_model = DepthModel(
            config.models.depth
        )
        self.object_detection_model = ObjectDetectionModel(
            config.models.object_detection
        )
        self.segmentation_model = SegmentationModel(
            config.models.segmentation
        )
        self.config = config

    def __call__(self, image: Image.Image) -> Image.Image:
        """Call the pipeline on an image."""
        return self.predict(image)

    def predict(self, image: Image.Image) -> Image.Image:
        """
        Run inference on a single image.
        
        :param image: Input PIL Image
        :return: Segmentation masks
        """
        return run_inference(
            image,
            self.saliency_model,
            self.depth_model,
            self.object_detection_model,
            self.segmentation_model,
            self.config
        )
    
    def process_directory(
        self,
        input_img_dir: str,
        output_img_dir: str,
        output_mask_dir: str
    ) -> None:
        """
        Run inference on all images in a directory.
        
        :param input_img_dir: Directory containing input images
        :param output_img_dir: Directory to save combined images
        :param output_mask_dir: Directory to save masks
        """
        # Ensure output directories exist
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        
        input_imgs = os.listdir(input_img_dir)
        total_images = len(input_imgs)

        for idx, input_img in enumerate(tqdm(input_imgs, desc="Processing images", ncols=100)):
            if not input_img.lower().endswith(('.jpg', '.png')): 
                continue

            logging.info(
                f'\nProcessing image {idx + 1}/{total_images}: {input_img}'
            )
            
            # Read input image
            base_name = input_img.split('.')[0]
            img_path = os.path.join(input_img_dir, input_img)
            in_img = load_image(img_path)

            # Get segmentation masks
            masks = self(in_img)

            # Get combined mask
            combined_mask = get_mask_img(masks)

            # Save output mask
            mask_name = base_name + '.png'
            output_path = os.path.join(output_mask_dir, mask_name)
            output_mask = Image.fromarray((combined_mask * 255).astype(np.uint8))
            output_mask.save(output_path)
            logging.info(f'Saved mask: {output_path}')

            # Save combined images
            combined_image = combine_image_with_mask(in_img, output_mask)
            combined_image.save(os.path.join(output_img_dir, base_name + '.png'))
            logging.info(
                f'Saved combined image: {os.path.join(output_img_dir, base_name + ".png")}'
            )