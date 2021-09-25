"""This module is used to damage an image."""
import os
import random
from typing import Tuple

import cv2 as cv
import numpy as np

IMAGE_PERCENTAGE_TO_DESTROY_LOW = 0.2
IMAGE_PERCENTAGE_TO_DESTROY_HIGH = 0.3
WHITE_VALUE = 255
BLACK_VALUE = 0
BLACK_VALUE_BRG = (0, 0, 0)


def damage_image(image_path: str,
                 noise_mean_value: float,
                 noise_variance: float) -> None:
    """Damage an image.

    This is done by adding Gaussian noise
    and destroying a portion of it.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image {image_path} not found')

    print(f'Damaging image {image_path}...')

    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    noisy_image = _add_noise(image, noise_mean_value, noise_variance)

    missing_part_points = _calculate_missing_part_points(image.shape)
    mask_image = _create_mask_image(missing_part_points, image.shape)

    damaged_image = cv.rectangle(noisy_image,
                                 missing_part_points[0],
                                 missing_part_points[1],
                                 BLACK_VALUE_BRG,
                                 -1)

    damaged_image_path = image_path.replace('.', '-damaged.')
    mask_image_path = image_path.replace('.', '-mask.')

    cv.imwrite(damaged_image_path, damaged_image)
    cv.imwrite(mask_image_path, mask_image)

    print('Done!')


def _calculate_missing_part_points(
        shape: Tuple[int, ...]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Create the coordinates of the destroyed portion of the image."""
    missing_part_height = random.randint(
        int(shape[0] * IMAGE_PERCENTAGE_TO_DESTROY_LOW),
        int(shape[0] * IMAGE_PERCENTAGE_TO_DESTROY_HIGH))
    missing_part_width = random.randint(
        int(shape[1] * IMAGE_PERCENTAGE_TO_DESTROY_LOW),
        int(shape[1] * IMAGE_PERCENTAGE_TO_DESTROY_HIGH))

    missing_part_start_y = random.randint(0, shape[0] - missing_part_height)
    missing_part_start_x = random.randint(0, shape[1] - missing_part_width)

    missing_part_start_point = (missing_part_start_x, missing_part_start_y)
    missing_part_end_point = (missing_part_start_x + missing_part_width - 1,
                              missing_part_start_y + missing_part_height - 1)

    missing_part_points = (missing_part_start_point, missing_part_end_point)

    return missing_part_points


def _create_mask_image(
        missing_part_points: Tuple[Tuple[int, int], Tuple[int, int]],
        shape: Tuple[int, ...]) -> np.ndarray:
    """Create the mask image.

    The mask image indicates which pixels have been damaged.
    """
    white_image = np.full(shape, WHITE_VALUE, dtype=np.uint8)
    mask_image = cv.rectangle(white_image, missing_part_points[0],
                              missing_part_points[1], BLACK_VALUE_BRG, -1)

    return mask_image


def _add_noise(image: np.ndarray,
               noise_mean_value: float,
               noise_variance: float) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.normal(noise_mean_value, noise_variance, image.shape)
    image_norm = cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    noisy_image_norm = image_norm + noise
    noisy_image = cv.normalize(noisy_image_norm, None, BLACK_VALUE,
                               WHITE_VALUE, cv.NORM_MINMAX, cv.CV_8U)

    return noisy_image
