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
    n_args = len(args)

    if n_args != N_ARGS_EXPECTED:
        raise TypeError('Wrong number of parameters:'
                        f'expected {N_ARGS_EXPECTED}, received {n_args}')

    image_path = args[0]
    noise_mean_value = float(args[1])
    noise_variance = float(args[2])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image {image_path} not found')

    print(f'Damaging image {image_path}...')

    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    damaged_image, mask_image = damage_image(image, noise_mean_value,
                                             noise_variance)

    damaged_image_path = image_path.replace('.', '-damaged.')
    mask_image_path = image_path.replace('.', '-mask.')

    cv.imwrite(damaged_image_path, damaged_image)
    cv.imwrite(mask_image_path, mask_image)

    print('Done!')


def damage_image(image: np.ndarray, noise_mean_value: float,
                 noise_variance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Damage an image
    by adding Gaussian noise and destroying a portion of it.
    """
    noisy_image = add_noise(image, noise_mean_value, noise_variance)

    missing_part_points = calculate_missing_part_points(image.shape)
    mask_image = create_mask_image(missing_part_points, image.shape)

    damaged_image = cv.rectangle(noisy_image, missing_part_points[0],
                                 missing_part_points[1], BLACK_VALUE_BRG, -1)

    return damaged_image, mask_image


def calculate_missing_part_points(
        shape: Tuple[int, ...]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Create the 4 coordinates
    of the destroyed portion of the image.
    """
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


def create_mask_image(
        missing_part_points: Tuple[Tuple[int, int], Tuple[int, int]],
        shape: Tuple[int, ...]) -> np.ndarray:
    """Create the mask image
    which indicates which pixels have been damaged.
    """
    white_image = np.full(shape, WHITE_VALUE, dtype=np.uint8)
    mask_image = cv.rectangle(white_image, missing_part_points[0],
                              missing_part_points[1], BLACK_VALUE_BRG, -1)

    return mask_image


def add_noise(image: np.ndarray, noise_mean_value: float,
              noise_variance: float) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.normal(noise_mean_value, noise_variance, image.shape)
    image_norm = cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    noisy_image_norm = image_norm + noise
    noisy_image = cv.normalize(noisy_image_norm, None, BLACK_VALUE,
                               WHITE_VALUE, cv.NORM_MINMAX, cv.CV_8U)

    return noisy_image


if __name__ == '__main__':
    main(sys.argv[1:])
