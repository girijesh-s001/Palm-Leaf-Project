import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts an BGR/RGB image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_otsu_threshold(gray_image: np.ndarray, invert: bool = True) -> np.ndarray:
    """Applies Otsu's thresholding to convert a grayscale image into a binary image.

    If invert=True (default for dark text on light palm leaf), character strokes become 255 (foreground)
    and background becomes 0.
    """
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary_img = cv2.threshold(
        gray_image, 0, 255, flag + cv2.THRESH_OTSU
    )
    return binary_img


def resize_image(image: np.ndarray, target_size: tuple = (64, 64)) -> np.ndarray:
    """Resizes image to the target size (width, height)."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalizes pixel values to [0.0, 1.0] and expands dimensions to (H, W, 1)."""
    normalized = image.astype(np.float32) / 255.0
    if len(normalized.shape) == 2:
        normalized = np.expand_dims(normalized, axis=-1)
    return normalized


def preprocess_pipeline(
    image: np.ndarray, target_size: tuple = (64, 64), invert: bool = True
) -> np.ndarray:
    """Complete image preprocessing pipeline:

    1. RGB/BGR to Grayscale
    2. Otsu Thresholding
    3. Resize to 64x64
    4. Normalization [0.0, 1.0]
    """
    gray = convert_to_grayscale(image)
    binary = apply_otsu_threshold(gray, invert=invert)
    resized = resize_image(binary, target_size=target_size)
    normalized = normalize_image(resized)
    return normalized
