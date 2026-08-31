import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_otsu_threshold(gray_image: np.ndarray, invert: bool = True) -> np.ndarray:
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary_img = cv2.threshold(gray_image, 0, 255, flag + cv2.THRESH_OTSU)
    return binary_img


def resize_image(image: np.ndarray, target_size: tuple = (64, 64)) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    if normalized.ndim == 2:
        normalized = np.expand_dims(normalized, axis=-1)
    return normalized


def preprocess_pipeline(image: np.ndarray, target_size: tuple = (64, 64), invert: bool = True) -> np.ndarray:
    """Preprocesses a raw crop into a normalized tensor ready for the CNN."""
    gray = convert_to_grayscale(image)
    binary = apply_otsu_threshold(gray, invert=invert)
    resized = resize_image(binary, target_size=target_size)
    return normalize_image(resized)

