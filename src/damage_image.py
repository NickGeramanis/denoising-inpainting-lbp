"""This module is used to damage images."""
import os
import random
import sys
from typing import List, Tuple

import cv2 as cv
import numpy as np

N_ARGS_EXPECTED = 3
IMAGE_PERCENTAGE_TO_DESTROY_LOW = 0.2
IMAGE_PERCENTAGE_TO_DESTROY_HIGH = 0.3
WHITE_VALUE = 255
BLACK_VALUE = 0
BLACK_VALUE_BRG = (0, 0, 0)


def main(args: List[str]) -> None:
    """Main function."""
    n_args = len(args)

    if n_args != N_ARGS_EXPECTED:
        raise TypeError('Wrong number of parameters:'
                        f'expected {N_ARGS_EXPECTED}, received {n_args}')

    path = args[0]
    noise_mean_value = float(args[1])
    noise_variance = float(args[2])

    if not os.path.exists(path):
        raise FileNotFoundError(f'Image {path} not found')

    print(f'Damaging image {path}...')

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)

    damaged_image, mask_image = damage_image(image, noise_mean_value,
                                             noise_variance)

    path_tokens = path.split('.')
    path_without_extension = path_tokens[0]
    extension = path_tokens[1]

    cv.imwrite(f'{path_without_extension}-damaged.{extension}', damaged_image)
    cv.imwrite(f'{path_without_extension}-mask.{extension}', mask_image)

    print('Done!')


def damage_image(image: np.ndarray, noise_mean_value: float,
                 noise_variance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Damage an image (add noise and destroy a portion of it)."""
    noisy_image = add_noise(image, noise_mean_value, noise_variance)

    image_height, image_width = image.shape
    missing_part_start_point, missing_part_end_point, mask_image = (
        create_mask_image(image_height, image_width))

    damaged_image = cv.rectangle(noisy_image, missing_part_start_point,
                                 missing_part_end_point, BLACK_VALUE_BRG, -1)

    return damaged_image, mask_image


def create_mask_image(
        height: int,
        width: int) -> Tuple[Tuple[int, int], Tuple[int, int], np.ndarray]:
    """Create the mask image (indicates which pixels have been damaged)."""
    missing_part_height = random.randint(
        int(height * IMAGE_PERCENTAGE_TO_DESTROY_LOW),
        int(height * IMAGE_PERCENTAGE_TO_DESTROY_HIGH))
    missing_part_width = random.randint(
        int(width * IMAGE_PERCENTAGE_TO_DESTROY_LOW),
        int(width * IMAGE_PERCENTAGE_TO_DESTROY_HIGH))
    missing_part_start_x = random.randint(0, width - missing_part_width)
    missing_part_start_y = random.randint(0, height - missing_part_height)
    missing_part_start_point = (missing_part_start_x, missing_part_start_y)
    missing_part_end_point = (missing_part_start_x + missing_part_width - 1,
                              missing_part_start_y + missing_part_height - 1)

    white_image = np.full((height, width), WHITE_VALUE, dtype=np.uint8)
    mask_image = cv.rectangle(white_image, missing_part_start_point,
                              missing_part_end_point, BLACK_VALUE_BRG, -1)

    return missing_part_start_point, missing_part_end_point, mask_image


def add_noise(image: np.ndarray, mean_value: float,
              variance: float) -> np.ndarray:
    """Add Gaussian noise to an image."""
    image_height, image_width = image.shape

    noise = np.random.normal(mean_value, variance, (image_height, image_width))
    image_norm = cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    noisy_image_norm = image_norm + noise
    noisy_image = cv.normalize(noisy_image_norm, None, BLACK_VALUE,
                               WHITE_VALUE, cv.NORM_MINMAX, cv.CV_8U)

    return noisy_image


if __name__ == '__main__':
    main(sys.argv[1:])
