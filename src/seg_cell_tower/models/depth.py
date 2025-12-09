from PIL import Image
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
import torch.nn.functional as F
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from typing import Any, Dict, Tuple


class DepthModel(object):

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DepthModel.

        :param config: Configuration containing model parameters.
            - ckpt: Path to the model checkpoint.
            - device: Device to run the model on (e.g., 'cpu', 'cuda').
        """
        self.model = DepthAnything.from_pretrained(config.ckpt).to(config.device)
        self.model.eval()

        self.device = config.device

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.predict(image)

    def predict(self, image: Image.Image, grayscale: bool = True) -> np.ndarray:
        """
        Predict the depth of the image.

        :param image: Input image.
        :param grayscale: Whether to return the depth image in grayscale.
        :return: Depth map of the image.
        """
        w, h = image.size
        image = self.preprocess(image)

        with torch.no_grad():
            depth = self.model(image)

        depth = DepthModel.postprocess(depth, (h, w), grayscale)

        return depth

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the input image.

        :param image: Input image.
        :return: Preprocessed image.
        """

        self.transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        image = np.asarray(image) / 255.0
        image = self.transform({"image": image})["image"]

        return torch.from_numpy(image).unsqueeze(0).to(self.device)

    @staticmethod
    def postprocess(
        depth: torch.Tensor, size: Tuple[int, int], grayscale: bool
    ) -> np.ndarray:
        """
        Postprocess the depth map.

        :param depth: Depth map.
        :param grayscale: Whether to return the depth image in grayscale.
        :return: Postprocessed depth map.
        """
        depth = F.interpolate(
            depth[None], size, mode="bilinear", align_corners=False)[0, 0]

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.cvtColor(
                cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB
            )

        return depth
