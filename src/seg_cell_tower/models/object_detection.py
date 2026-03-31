from typing import Dict, Any
import torch
from PIL import Image
import numpy as np
from groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T


class ObjectDetectionModel(object):

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ObjectDetectionModel.

        :param config: Configuration containing model parameters.
            - ckpt: Path to the model checkpoint.
            - config_path: Path to the model configuration file.
            - text_prompt: Text prompt for object detection.
            - obj_det_box: Threshold for box predictions.
            - obj_det_text: Threshold for text predictions.
        """
        self.model = load_model(
            config.config_path,
            config.ckpt,
            device=config.device
        )
        self.config = config

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.predict(image)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict the bounding boxes for the image.

        :param image: Input image.
        :return: Bounding boxes in xyxy format.
        """
        h, w, _ = image.shape
        transformed_img = ObjectDetectionModel.preprocess(image)

        boxes, logits, phrases = predict(
            model=self.model,
            image=transformed_img,
            caption=self.config.text_prompt,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold
        )

        boxes_xyxy = ObjectDetectionModel.post_process_result(
            source_h=h,
            source_w=w,
            boxes=boxes
        )

        return {"boxes": boxes_xyxy, "scores": logits.numpy()}

    @staticmethod
    def preprocess(img: Image.Image) -> torch.Tensor:
        """
        Transform the input image for the model.

        :param img: Input image as a numpy array.
        :return: Transformed image as a torch tensor.
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        img = Image.fromarray(img)
        image_transformed, _ = transform(img, None)

        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor
    ) -> np.ndarray:
        """
        Post-process the model output.

        :param source_h: Height of the source image.
        :param source_w: Width of the source image.
        :param boxes: Predicted bounding boxes.
        :param logits: Predicted logits.
        :return: Processed bounding boxes in xyxy format.
        """
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        return xyxy
