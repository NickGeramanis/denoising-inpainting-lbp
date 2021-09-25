import math
import os

import cv2 as cv
import numpy as np
import pytest

from denoising_inpainting_lbp import denoising_inpainting


def test_image_name():
    with pytest.raises(FileNotFoundError):
        denoising_inpainting.denoise_inpaint('img.png', 'img.png', 1, 1.5, 1.5)


def test_calculate_energy():
    path = os.path.dirname(__file__)
    observed_image = cv.imread(path + '/house-damaged.png',
                               cv.IMREAD_GRAYSCALE)
    mask_image = cv.imread(path + '/house-mask.png', cv.IMREAD_GRAYSCALE)
    labeled_image = np.copy(observed_image)
    lambda_ = 5
    maximum_smoothness_penalty = math.inf

    energy = denoising_inpainting._calculate_energy(observed_image,
                                                    labeled_image,
                                                    mask_image,
                                                    lambda_,
                                                    maximum_smoothness_penalty)

    assert energy == 686805375
