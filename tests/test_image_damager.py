import os

import cv2 as cv
import numpy as np
import pytest

from denoising_inpainting_lbp import image_damager

image = np.random.randint(256, size=(256, 144))
noise_mean_value = 0
noise_variance = 0.1


def test_wrong_image_path():
    with pytest.raises(FileNotFoundError):
        image_damager.damage_image('img.png', noise_mean_value, noise_variance)


def test_damaged_mask_images_creation():
    path = os.path.dirname(__file__)
    image_damager.damage_image(path + '/boat.png',
                               noise_mean_value,
                               noise_variance)

    assert os.path.exists(path + '/boat-damaged.png')
    assert os.path.exists(path + '/boat-mask.png')


def test_noisy_image_shape():
    noisy_image = image_damager._add_noise(image,
                                           noise_mean_value,
                                           noise_variance)

    assert image.shape == noisy_image.shape


def test_noisy_image_pixels():
    # 99.7% success rate due to random noise (tolerance is 3SD)
    noisy_image = image_damager._add_noise(image,
                                           noise_mean_value,
                                           noise_variance)
    noisy_image_norm = cv.normalize(noisy_image,
                                    None,
                                    0,
                                    1,
                                    cv.NORM_MINMAX,
                                    cv.CV_32F)
    image_norm = cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    assert np.allclose(image_norm, noisy_image_norm, atol=0.94868329805)


def test_missing_part_points():
    missing_part_points = image_damager._calculate_missing_part_points(
        image.shape)

    assert (0
            <= missing_part_points[0][0]
            <= missing_part_points[1][0]
            <= image.shape[1] - 1)
    assert (0
            <= missing_part_points[0][1]
            <= missing_part_points[1][1]
            <= image.shape[0] - 1)


def test_mask_image_shape():
    missing_part_points = image_damager._calculate_missing_part_points(
        image.shape)
    mask_image = image_damager._create_mask_image(missing_part_points,
                                                  image.shape)

    assert image.shape == mask_image.shape


def test_mask_image_pixels():
    missing_part_points = image_damager._calculate_missing_part_points(
        image.shape)
    mask_image = image_damager._create_mask_image(missing_part_points,
                                                  image.shape)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            row_in_missing_part = (missing_part_points[0][1]
                                   <= row
                                   <= missing_part_points[1][1])
            col_in_missing_part = (missing_part_points[0][0]
                                   <= col
                                   <= missing_part_points[1][0])
            if row_in_missing_part and col_in_missing_part:
                assert mask_image[row, col] == image_damager.BLACK_VALUE
            else:
                assert mask_image[row, col] == image_damager.WHITE_VALUE
