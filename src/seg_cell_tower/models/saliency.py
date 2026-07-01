from PIL import Image
from transparent_background import Remover

from ..config import SaliencyConfig
from ..logging import get_logger

logger = get_logger(__name__)


class SaliencyDetectionModel:
    def __init__(self, config: SaliencyConfig):
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
            image = self.model.process(image, type="white")
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            raise

        return image
