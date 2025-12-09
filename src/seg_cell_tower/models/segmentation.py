from typing import Dict, Any
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


class SegmentationModel(object):

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SegmentationModel.

        :param config: Configuration containing model parameters.
            - ckpt: Path to the model checkpoint.
            - model_type: Type of the model.
            - device: Device to run the model on (e.g., 'cpu', 'cuda').
        """
        sam = sam_model_registry[config.model_type](
            checkpoint=config.ckpt
        ).to(device=config.device)
        self.model = SamPredictor(sam)

        self.device = config.device

    def __call__(
        self, image: Image.Image, prompts: np.ndarray = None
    ) -> np.ndarray:
        return self.predict(image, prompts)

    def predict(
        self, image: Image.Image, prompts: np.ndarray = None
    ) -> np.ndarray:
        """
        Predict the segmentation masks for the image.

        :param image: Input image.
        :param prompts: Input prompts for segmentation.
        :return: Segmentation masks.
        """
        image = np.asarray(image)
        self.model.set_image(image)

        result_masks = []
        for box in prompts:
            masks, scores, logits = self.model.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])

        return np.array(result_masks)
