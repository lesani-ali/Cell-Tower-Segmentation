from PIL import Image
import numpy as np


def load_image(image_path: str) -> Image.Image:
    """Load an image from disk."""
    image = Image.open(image_path).convert("RGB")
    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save a numpy array as an image."""
    output_image = Image.fromarray(image)
    output_image.save(output_path)