import math
import os

import cv2 as cv
import numpy as np
import pytest

import denoising_inpainting


class TestDensoisingInpainting:
    def test_args_number(self):
        with pytest.raises(TypeError):
            args = ['image.png', 1]
            denoising_inpainting.main(args)

    def test_image_name(self):
        with pytest.raises(FileNotFoundError):
            args = ['image.png', 1, 2, 3, 4]
            denoising_inpainting.main(args)

    def test_calculate_energy(self):
        path = os.path.dirname(__file__) + '/../images/'
        observed_image = cv.imread(path + 'house-damaged.png',
                                   cv.IMREAD_GRAYSCALE)
        mask_image = cv.imread(path + 'house-mask.png', cv.IMREAD_GRAYSCALE)
        labeled_image = np.copy(observed_image)
        lambda_value = 5
        maximum_smoothness_penalty = math.inf

        energy = denoising_inpainting.calculate_energy(
            observed_image, labeled_image, mask_image, lambda_value,
            maximum_smoothness_penalty)

        assert energy == 686805375
