from typing import List, Optional, Union

import cv2
import numpy as np


def apply_brightness(image: np.ndarray, parameter: List[float]) -> np.ndarray:
    return np.clip(image + np.array(parameter), 0, 255).astype('uint8')


def apply_contrast(image: np.ndarray, parameter: List[float]) -> np.ndarray:
    return np.clip(parameter[0] * image, 0, 255).astype('uint8')


def apply_tint(image: np.ndarray, parameter: List[float]) -> np.ndarray:
    channel_map = {'blue': 0, 'green': 1, 'red': 2}
    channel = channel_map[parameter[0].lower()]
    alpha = 1 + parameter[1] / 100
    new_image = image.copy()
    new_image[:, :, channel] = np.clip(image[:, :, channel] * alpha, 0, 255)
    return new_image.astype('uint8')


def apply_gamma_correction(image: np.ndarray, parameter: List[float]) -> np.ndarray:
    invGamma = 1.0 / parameter[0]
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_blur(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    kernel = (1 / 256) * np.array(
        [[1, 4, 6, 4, 1],
         [4, 16, 24, 16, 4],
         [6, 24, 36, 24, 6],
         [4, 16, 24, 16, 4],
         [1, 4, 6, 4, 1]]
    )
    return cv2.filter2D(image, -1, kernel)


def apply_edges(image: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [[-1, -1, -1],
         [-1, 8, -1],
         [-1, -1, -1]]
    )
    return cv2.filter2D(image, -1, kernel)


def apply_sharpening(image: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [[-1, -1, -1],
         [-1, 9, -1],
         [-1, -1, -1]]
    )
    return cv2.filter2D(image, -1, kernel)


def apply_hflip(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=1)

def apply_vflip(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=0)

def apply_rotation(image: np.ndarray, parameter: List[float]) -> np.ndarray:
    height, width = image.shape[:2]
    m = cv2.getRotationMatrix2D((width / 2, height / 2), parameter[0], 1)
    return cv2.warpAffine(image, m, (width, height))


def apply_translation(image: np.ndarray, parameter: List[float]) -> np.ndarray:
    height, width = image.shape[:2]
    m = np.float32([[1, 0, parameter[0]], [0, 1, parameter[1]]])
    return cv2.warpAffine(image, m, (width, height))


def apply_resize(image: np.ndarray, parameter: List[int]) -> np.ndarray:
    return cv2.resize(image, (parameter[0], parameter[1]))


def apply(image: np.ndarray, augmentation: str, parameter: Optional[List] = None) -> np.ndarray:
    """
    Apply an augmentation to the image.

    :param image: The input image.
    :param augmentation: The type of augmentation to apply.
    :param parameter: Parameters required for the augmentation.
    :return: The augmented image.
    """
    augmentation = augmentation.lower()
    if augmentation == 'brightness':
        return apply_brightness(image, parameter)
    elif augmentation == 'contrast':
        return apply_contrast(image, parameter)
    elif augmentation == 'tint':
        return apply_tint(image, parameter)
    elif augmentation == 'gammacorrection':
        return apply_gamma_correction(image, parameter)
    elif augmentation == 'blur':
        return apply_blur(image)
    elif augmentation == 'gaussianblur':
        return apply_gaussian_blur(image)
    elif augmentation == 'edges':
        return apply_edges(image)
    elif augmentation == 'sharpening':
        return apply_sharpening(image)
    elif augmentation == 'hflip':
        return apply_hflip(image)
    elif augmentation == 'vflip':
        return apply_vflip(image)
    elif augmentation == 'rotation':
        return apply_rotation(image, parameter)
    elif augmentation == 'translation':
        return apply_translation(image, parameter)
    elif augmentation == 'resize':
        return apply_resize(image, parameter)
    else:
        return image
