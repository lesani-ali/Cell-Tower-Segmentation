from PIL import Image
import numpy as np


def combine_image_with_mask(
    image: Image.Image,
    combined_mask: np.ndarray
) -> Image.Image:
    """
    Combine original image with mask overlay.
    
    :param image: Original RGB image
    :param mask: Mask as numpy array
    :return: Combined RGBA image
    """
    image = image.convert('RGBA')
    combined_image = Image.alpha_composite(image, combined_mask)
    return combined_image


def get_mask_img(masks, random_color=False):
    if len(masks) == 0:
        return
    h, w = masks.shape[-2:]
    rgb_mask = np.zeros((h, w, 4))

    if random_color:
        for mask in masks:
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            rgb_mask[mask] = color_mask
    else:
        for mask in masks:
            color_mask = np.array([30/255, 144/255, 255/255, 0.35])
            rgb_mask[mask] = color_mask
    return rgb_mask