import os
import sys

import cv2 as cv
import numpy as np

from denoising_inpainting_lbp.markov_random_field import MarkovRandomField


def test_calculate_energy():
    path = os.path.dirname(__file__)
    observed_image = cv.imread(path + '/house-damaged.png',
                               cv.IMREAD_GRAYSCALE)
    mask_image = cv.imread(path + '/house-mask.png', cv.IMREAD_GRAYSCALE)
    lambda_ = 5
    max_smoothness_penalty = sys.maxsize
    mrf = MarkovRandomField(observed_image.astype(np.int64),
                            mask_image.astype(np.int64),
                            lambda_,
                            max_smoothness_penalty)

    energy = mrf.calculate_energy()

    assert energy == 686805375
