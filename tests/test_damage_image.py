import math

import cv2 as cv
import numpy as np
import pytest

from src import damage_image


class TestDamageImage:
    image = np.random.randint(256, size=(256, 144))
    noise_mean_value = 0
    noise_variance = 0.1

    def test_wrong_number_of_args(self):
        with pytest.raises(TypeError):
            args = ['image.png', 1]
            damage_image.main(args)

    def test_wrong_image_path(self):
        with pytest.raises(FileNotFoundError):
            args = ['image.png', 1, 2]
            damage_image.main(args)

    def test_noisy_image_shape(self):
        noisy_image = damage_image.add_noise(self.image, self.noise_mean_value,
                                             self.noise_variance)

        assert self.image.shape == noisy_image.shape

    def test_noisy_image_pixel_values(self):
        # 99.7% success rate due to random noise (tolerance is 3SD)
        noisy_image = damage_image.add_noise(self.image, self.noise_mean_value,
                                             self.noise_variance)
        image_norm = cv.normalize(self.image, None, 0, 1, cv.NORM_MINMAX,
                                  cv.CV_32F)
        noisy_image_norm = cv.normalize(noisy_image, None, 0, 1,
                                        cv.NORM_MINMAX, cv.CV_32F)

        standard_deviation = math.sqrt(self.noise_variance)
        assert np.allclose(image_norm, noisy_image_norm,
                           atol=3 * standard_deviation)

    def test_calculate_missing_part_points(self):
        missing_part_points = damage_image.calculate_missing_part_points(
            self.image.shape)

        assert (0 <= missing_part_points[0][0] <= missing_part_points[1][0]
                <= self.image.shape[1] - 1)
        assert (0 <= missing_part_points[0][1] <= missing_part_points[1][1]
                <= self.image.shape[0] - 1)

    def test_mask_image_shape(self):
        missing_part_points = damage_image.calculate_missing_part_points(
            self.image.shape)
        mask_image = damage_image.create_mask_image(missing_part_points,
                                                    self.image.shape)

        assert self.image.shape == mask_image.shape

    def test_mask_image_pixel_values(self):
        missing_part_points = damage_image.calculate_missing_part_points(
            self.image.shape)
        mask_image = damage_image.create_mask_image(missing_part_points,
                                                    self.image.shape)

        for row in range(self.image.shape[0]):
            for col in range(self.image.shape[1]):
                row_in_missing_par = (missing_part_points[0][1] <= row
                                      <= missing_part_points[1][1])
                col_in_missing_par = (missing_part_points[0][0] <= col
                                      <= missing_part_points[1][0])
                if row_in_missing_par and col_in_missing_par:
                    assert mask_image[row, col] == damage_image.BLACK_VALUE
                else:
                    assert mask_image[row, col] == damage_image.WHITE_VALUE
