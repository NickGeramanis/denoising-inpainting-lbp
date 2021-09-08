import math

import cv2 as cv
import numpy as np
import pytest

from denoising_inpainting_lbp import damage_image


class TestDamageImage:
    image_size = (256, 144)
    image = np.random.randint(256, size=image_size)
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

        assert self.image_size == noisy_image.shape

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

    def test_missing_part_points(self):
        missing_part_start_point, missing_part_end_point, _ = (
            damage_image.create_mask_image(self.image_size[0],
                                           self.image_size[1]))

        assert (0 <= missing_part_start_point[0] <= missing_part_end_point[
            0] <= self.image_size[1] - 1)
        assert (0 <= missing_part_start_point[1] <= missing_part_end_point[
            1] <= self.image_size[0] - 1)

    def test_mask_image_shape(self):
        _, _, mask_image = damage_image.create_mask_image(self.image_size[0],
                                                          self.image_size[1])

        assert self.image_size == mask_image.shape

    def test_mask_image_pixel_values(self):
        missing_part_start_point, missing_part_end_point, mask_image = (
            damage_image.create_mask_image(self.image_size[0],
                                           self.image_size[1]))

        for row_i in range(self.image_size[0]):
            for col_i in range(self.image_size[1]):
                if (missing_part_start_point[1] <= row_i <=
                    missing_part_end_point[1]) and (
                        missing_part_start_point[0] <= col_i <=
                        missing_part_end_point[0]):
                    assert mask_image[row_i][col_i] == damage_image.BLACK_VALUE
                else:
                    assert mask_image[row_i][col_i] == damage_image.WHITE_VALUE
