import logging
import math
import os
import sys
import time
from typing import List, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

N_LABELS = 256
N_ARGS_EXPECTED = 6

logger = logging.getLogger(__name__)
log_formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S')
file_handler = logging.FileHandler('lbp_info.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def main(args: List[str]) -> None:
    n_args = len(args)

    if n_args != N_ARGS_EXPECTED:
        raise TypeError('Wrong number of parameters: '
                        f'expected {N_ARGS_EXPECTED}, received {n_args}')

    image_path = args[0]
    mask_image_path = args[1]

    if not os.path.exists(image_path) or not os.path.exists(mask_image_path):
        raise FileNotFoundError('Required images not found')

    observed_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    mask_image = cv.imread(mask_image_path, cv.IMREAD_GRAYSCALE)

    n_iterations = int(args[2])
    lambda_ = float(args[3])
    max_smoothness_penalty = math.inf if args[4] == 'inf' else float(args[4])
    energy_lower_bound = float(args[5])

    labeled_image, energy, duration = min_sum(observed_image, mask_image,
                                              n_iterations,
                                              max_smoothness_penalty, lambda_)

    labeled_image_path = image_path.replace('.', '-labeled.')
    cv.imwrite(labeled_image_path, labeled_image)
    print(energy)
    print(energy_lower_bound)
    print(energy / energy_lower_bound)
    plt.plot(duration, energy / energy_lower_bound, 'ro-')
    plt.ylabel('Energy')
    plt.xlabel('Time (s)')
    plt.savefig('energy_vs_time.png')

    logger.info('Done!')


def min_sum(observed_image: np.ndarray, mask_image: np.ndarray,
            n_iterations: int, max_smoothness_penalty: float,
            lambda_: float,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute the min-sum algorithm (loopy belief propagation)
    to perform denoising and inpainting on an image.
    """
    logger.info('Running Loopy Belief Propagation algorithm (min-sum version) '
                'for %d iteration(s)...', n_iterations)

    starting_energy = calculate_energy(observed_image, observed_image,
                                       mask_image, lambda_,
                                       max_smoothness_penalty)
    logger.info('Starting Energy = %f', starting_energy)

    image_height, image_width = observed_image.shape

    energy = np.empty(n_iterations+1, dtype=np.uint64)
    energy[0] = starting_energy
    duration = np.empty(n_iterations+1, dtype=np.float64)
    duration[0] = 0

    # Initialization of the messages,
    incoming_messages_right = np.zeros(
        (image_height, image_width, N_LABELS), dtype=np.uint64)
    incoming_messages_left = np.zeros(
        (image_height, image_width, N_LABELS), dtype=np.uint64)
    incoming_messages_down = np.zeros(
        (image_height, image_width, N_LABELS), dtype=np.uint64)
    incoming_messages_up = np.zeros(
        (image_height, image_width, N_LABELS), dtype=np.uint64)

    data_cost = init_data_cost(observed_image, mask_image)
    smoothness_cost = init_smoothness_cost(lambda_, max_smoothness_penalty)

    starting_time = time.time()
    # iteratively update incoming messages for each node
    for iteration in range(1, n_iterations + 1):
        logger.info('Iteration %d', iteration)
        # pass messages along rows
        for row in range(image_height):
            # forward pass (left to right)
            for column in range(image_width - 1):
                for label in range(N_LABELS):
                    incoming_messages_left[row, column + 1, label] = np.amin(
                        data_cost[row, column]
                        + smoothness_cost[label, :]
                        + incoming_messages_up[row, column, :]
                        + incoming_messages_down[row, column, :]
                        + incoming_messages_left[row, column, :])

            # backward pass (right to left)
            for column in range(image_width - 1, 0, -1):
                for label in range(N_LABELS):
                    incoming_messages_right[row, column - 1, label] = np.amin(
                        data_cost[row, column, :]
                        + smoothness_cost[label, :]
                        + incoming_messages_up[row, column, :]
                        + incoming_messages_down[row, column, :]
                        + incoming_messages_right[row, column, :])

        # pass messages along columns
        for column in range(image_width):
            # forward pass (top down)
            for row in range(image_height - 1):
                for label in range(N_LABELS):
                    incoming_messages_up[row + 1, column, label] = np.amin(
                        data_cost[row, column, :]
                        + smoothness_cost[label, :]
                        + incoming_messages_left[row, column, :]
                        + incoming_messages_right[row, column, :]
                        + incoming_messages_up[row, column, :])

            # backward pass (bottom up)
            for row in range(image_height - 1, 0, -1):
                for label in range(N_LABELS):
                    incoming_messages_down[row - 1, column, label] = np.amin(
                        data_cost[row, column, :]
                        + smoothness_cost[label, :]
                        + incoming_messages_left[row, column, :]
                        + incoming_messages_right[row, column, :]
                        + incoming_messages_down[row, column, :])

        belief = compute_belief(incoming_messages_right,
                                incoming_messages_left, incoming_messages_down,
                                incoming_messages_up, data_cost)

        labeled_image = recover_map(belief)

        energy[iteration] = calculate_energy(observed_image, labeled_image,
                                             mask_image, lambda_,
                                             max_smoothness_penalty)
        duration[iteration] = time.time() - starting_time
        logger.info('Energy = %f, time = %f secs',
                    energy[iteration], duration[iteration])

    return labeled_image, energy, duration


def init_data_cost(observed_image: np.ndarray,
                   mask_image: np.ndarray) -> np.ndarray:
    """Initialise the data cost array."""
    image_height, image_width = observed_image.shape
    data_cost = np.empty(
        (image_height, image_width, N_LABELS), dtype=np.uint64)
    for row in range(image_height):
        for column in range(image_width):
            if mask_image[row, column] > 0:
                data_cost[row, column] = (
                    (int(observed_image[row, column]) - np.arange(0, N_LABELS))
                    ** 2)
    return data_cost


def init_smoothness_cost(lambda_: float,
                         max_smoothness_penalty: float) -> np.ndarray:
    """Initialise the smoothness cost array."""
    smoothness_cost = np.empty((N_LABELS, N_LABELS), dtype=np.uint64)
    for label in range(N_LABELS):
        for label2 in range(N_LABELS):
            smoothness_cost[label, label2] = (lambda_
                                              * min(max_smoothness_penalty,
                                                    (label - label2) ** 2))
    return smoothness_cost


def compute_belief(incoming_messages_right: np.ndarray,
                   incoming_messages_left: np.ndarray,
                   incoming_messages_down: np.ndarray,
                   incoming_messages_up: np.ndarray,
                   data_cost: np.ndarray) -> np.ndarray:
    """Compute belief."""
    image_height, image_width, _ = data_cost.shape
    belief = np.empty((image_height, image_width, N_LABELS), dtype=np.uint64)
    for row in range(image_height):
        for column in range(image_width):
            belief[row, column, :] = (
                data_cost[row, column, :]
                + incoming_messages_left[row, column, :]
                + incoming_messages_right[row, column, :]
                + incoming_messages_up[row, column, :]
                + incoming_messages_down[row, column, :]
            )
    return belief


def recover_map(belief: np.ndarray) -> np.ndarray:
    """Recover MAP configuration."""
    image_height, image_width, _ = belief.shape
    labeled_image = np.empty((image_height, image_width), dtype=np.uint64)
    for row in range(image_height):
        for column in range(image_width):
            labeled_image[row, column] = np.argmin(belief[row, column])

    return labeled_image


def calculate_energy(observed_image: np.ndarray, labeled_image: np.ndarray,
                     mask_image: np.ndarray, lambda_: float,
                     max_smoothness_penalty: float) -> float:
    """Calculate the energy of an image:

    E = Ep + Es

    Ep = Sum(Dp(lp))
    Dp(lp) = 0,         if lp is damaged
    Dp(lp) = (lp-op)^2, else

    Es = lambda * Sum(min(Vmax, Vpq(lp,lq)))
    Vpq(lp,lq) = (lp-lq)^2
    """
    image_height, image_width = labeled_image.shape
    data_energy = 0
    smoothness_energy = 0.0
    for row in range(image_height):
        for column in range(image_width):
            if mask_image[row, column] > 0:
                data_energy += (int(observed_image[row, column])
                                - int(labeled_image[row, column])) ** 2

            if row < image_height - 1:
                label_difference = (int(labeled_image[row, column])
                                    - int(labeled_image[row + 1, column])) ** 2
                smoothness_energy += min(max_smoothness_penalty,
                                         label_difference)
            if column < image_width - 1:
                label_difference = (int(labeled_image[row, column])
                                    - int(labeled_image[row, column + 1])) ** 2
                smoothness_energy += min(max_smoothness_penalty,
                                         label_difference)

    smoothness_energy *= lambda_

    energy = data_energy + smoothness_energy
    return energy


if __name__ == '__main__':
    main(sys.argv[1:])
