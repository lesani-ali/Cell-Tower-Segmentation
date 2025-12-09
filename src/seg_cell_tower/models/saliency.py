from typing import Dict, Any
from PIL import Image
import logging
from transparent_background import Remover


class SaliencyDetectionModel(object):

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SaliencyDetectionModel.

        :param config: Configuration containing model parameters.
            - mode: Mode of the model (e.g., 'base').
            - device: Device to run the model on (e.g., 'cpu', 'cuda').
        """
        self.model = Remover(mode=config.mode, jit=False, device=config.device)

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.predict(image)

    def predict(self, image: Image.Image) -> Image.Image:
        """
        Predict the saliency of the image and remove the background.

        :param image: Input image.
        :return: Image with background removed.
        """
        try:
            image = self.model.process(image, type='white')
        except Exception as e:
            logging.error(f"Error removing background: {e}")
            raise

        return image
