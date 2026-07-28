from PIL import Image


def load_image(image_path: str) -> Image.Image:
    """Load an image from disk."""
    image = Image.open(image_path).convert("RGB")
    return image
