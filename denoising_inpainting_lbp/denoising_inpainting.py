"""This module is used to perform denoising & inpainting on an image."""
import logging
import os
import sys
import time
from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from denoising_inpainting_lbp.markov_random_field import MarkovRandomField

logger = logging.getLogger(__name__)
log_formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S')

file_handler = logging.FileHandler('lbp.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)


# pylint: disable-next=too-many-arguments
def denoise_inpaint(image_path: str,
                    mask_image_path: str,
                    n_iterations: int,
                    lambda_: int,
                    energy_lower_bound: int,
                    max_smoothness_penalty: int = sys.maxsize) -> None:
    """Denoise & inpaint a damaged image using the LBP algorithm."""
    if not os.path.exists(image_path) or not os.path.exists(mask_image_path):
        raise FileNotFoundError('Required images not found')

    observed_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    mask_image = cv.imread(mask_image_path, cv.IMREAD_GRAYSCALE)

    labeled_image, energy, duration = _loopy_belief_propagation(
        observed_image,
        mask_image,
        n_iterations,
        max_smoothness_penalty,
        lambda_)

    labeled_image_path = image_path.replace('.', '-labeled.')
    cv.imwrite(labeled_image_path, labeled_image)

    plt.plot(duration, energy / energy_lower_bound, 'ro-')
    plt.ylabel('Energy')
    plt.xlabel('Time (s)')
    plt.savefig('energy_vs_time.png')

    logger.info('Done!')


def _loopy_belief_propagation(
        observed_image: np.ndarray,
        mask_image: np.ndarray,
        n_iterations: int,
        max_smoothness_penalty: int,
        lambda_: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute the Loopy Belief Propagation (LBP) algorithm (min-sum)."""
    logger.info('Running Loopy Belief Propagation algorithm (min-sum) '
                'for %d iteration(s)...', n_iterations)

    mrf = MarkovRandomField(observed_image.astype(np.int64),
                            mask_image.astype(np.int64),
                            lambda_,
                            max_smoothness_penalty)

    starting_energy = mrf.calculate_energy()
    logger.info('Starting Energy = %d', starting_energy)

    energy = np.empty(n_iterations + 1, dtype=np.int64)
    duration = np.empty(n_iterations + 1, dtype=np.float64)
    energy[0] = starting_energy
    duration[0] = 0

    starting_time = time.time()
    # Iteratively update incoming messages for each node.
    for iteration in range(1, n_iterations + 1):
        logger.info('Iteration %d', iteration)

        mrf.calculate_messages()
        mrf.recover_map()
        energy[iteration] = mrf.calculate_energy()
        duration[iteration] = time.time() - starting_time

        logger.info('Energy = %d, time = %f secs',
                    energy[iteration], duration[iteration])

    return mrf.retrieve_labels(), energy, duration
