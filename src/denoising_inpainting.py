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
N_ARGS_EXPECTED = 5

logger = logging.getLogger(__name__)
log_formatter = logging.Formatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s')
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

    path = args[0]
    path_tokens = path.split('.')
    path_without_extension = path_tokens[0]
    extension = path_tokens[1]

    damaged_image_exists = os.path.exists(
        f'{path_without_extension}-damaged.{extension}')
    mask_image_exists = os.path.exists(
        f'{path_without_extension}-mask.{extension}')
    if not damaged_image_exists or not mask_image_exists:
        raise FileNotFoundError('Required images not found')

    logger.info('Running loopy belief propagation...')

    observed_image = cv.imread(
        f'{path_without_extension}-damaged.{extension}',
        cv.IMREAD_GRAYSCALE)
    mask_image = cv.imread(f'{path_without_extension}-mask.{extension}',
                           cv.IMREAD_GRAYSCALE)

    max_iterations = int(args[1])
    lambda_value = int(args[2])
    maximum_smoothness_penalty = math.inf if args[3] == 'inf' else int(args[3])
    energy_lower_bound = float(sys.argv[4])

    labeled_image, energy, duration = min_sum(observed_image, mask_image,
                                              max_iterations, lambda_value,
                                              maximum_smoothness_penalty)

    cv.imwrite(f'{path_without_extension}-labeled.{extension}',
               labeled_image)

    plt.plot(duration, energy / energy_lower_bound, 'ro-')
    plt.ylabel('Energy')
    plt.xlabel('Time (s)')
    plt.savefig('energy_vs_time.png')

    logger.info('Done!')


def min_sum(observed_image: np.ndarray, mask_image: np.ndarray,
            max_iterations: int, lambda_value: int,
            max_smoothness_penalty: float) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_height, image_width = observed_image.shape

    energy = np.empty(max_iterations, dtype=np.uint64)
    duration = np.empty(max_iterations, dtype=np.float64)

    labeled_image = np.copy(observed_image)

    logger.info('Starting Energy')
    calculate_energy(observed_image, labeled_image, mask_image, lambda_value,
                     max_smoothness_penalty)

    belief = np.empty((image_height, image_width, N_LABELS),
                      dtype=np.uint64)

    # Initialization of the messages,
    incoming_messages_right = np.zeros(
        (image_height, image_width, N_LABELS), dtype=np.uint64)
    incoming_messages_left = np.zeros(
        (image_height, image_width, N_LABELS), dtype=np.uint64)
    incoming_messages_down = np.zeros(
        (image_height, image_width, N_LABELS), dtype=np.uint64)
    incoming_messages_up = np.zeros(
        (image_height, image_width, N_LABELS), dtype=np.uint64)

    data_cost = np.empty((image_height, image_width, N_LABELS),
                         dtype=np.uint64)
    for row in range(image_height):
        for column in range(image_width):
            if mask_image[row, column] > 0:
                data_cost[row, column] = (
                        (int(observed_image[row, column])
                         - np.arange(0, N_LABELS)) ** 2)

    smoothness_cost = np.empty((N_LABELS, N_LABELS),
                               dtype=np.uint64)
    for label in range(N_LABELS):
        for label2 in range(N_LABELS):
            smoothness_cost[label, label2] = (lambda_value
                                              * min(max_smoothness_penalty,
                                                    (label - label2) ** 2))

    logger.info('Running Loopy Belief Propagation algorithm(min-sum version) '
                f'for {max_iterations} iterations...')

    starting_time = time.time()

    # iteratively update incoming messages for each node
    for iteration in range(max_iterations):
        logger.info(f'Iteration {iteration}')
        # pass messages along rows
        for row in range(image_height):
            # forward pass (left to right)
            for column in range(image_width - 1):
                for label in range(N_LABELS):
                    outgoing_message_right = (
                            data_cost[row, column]
                            + smoothness_cost[label, :]
                            + incoming_messages_up[row, column, :]
                            + incoming_messages_down[row, column, :]
                            + incoming_messages_left[row, column, :])
                    incoming_messages_left[row, column + 1, label] = np.amin(
                        outgoing_message_right)

            # backward pass (right to left)
            for column in range(image_width - 1, 0, -1):
                for label in range(N_LABELS):
                    outgoing_message_left = (
                            data_cost[row, column, :]
                            + smoothness_cost[label, :]
                            + incoming_messages_up[row, column, :]
                            + incoming_messages_down[row, column, :]
                            + incoming_messages_right[row, column, :])
                    incoming_messages_right[row, column - 1, label] = np.amin(
                        outgoing_message_left)

        # pass messages along columns
        for column in range(image_width):
            # forward pass (top down)
            for row in range(image_height - 1):
                for label in range(N_LABELS):
                    outgoing_message_down = (
                            data_cost[row, column, :]
                            + smoothness_cost[label, :]
                            + incoming_messages_left[row, column, :]
                            + incoming_messages_right[row, column, :]
                            + incoming_messages_up[row, column, :])
                    incoming_messages_up[row + 1, column, label] = np.amin(
                        outgoing_message_down)

            # backward pass (bottom up)
            for row in range(image_height - 1, 0, -1):
                for label in range(N_LABELS):
                    outgoing_message_up = (
                            data_cost[row, column, :]
                            + smoothness_cost[label, :]
                            + incoming_messages_left[row, column, :]
                            + incoming_messages_right[row, column, :]
                            + incoming_messages_down[row, column, :])
                    incoming_messages_down[row - 1, column, label] = np.amin(
                        outgoing_message_up)

        # compute belief
        for row in range(image_height):
            for column in range(image_width):
                belief[row, column, :] = (
                        data_cost[row, column, :]
                        + incoming_messages_left[row, column, :]
                        + incoming_messages_right[row, column, :]
                        + incoming_messages_up[row, column, :]
                        + incoming_messages_down[row, column, :])

        # recover MAP configuration
        for row in range(image_height):
            for column in range(image_width):
                labeled_image[row, column] = np.argmin(belief[row, column])

        energy[iteration] = calculate_energy(observed_image, labeled_image,
                                             mask_image, lambda_value,
                                             max_smoothness_penalty)
        duration[iteration] = time.time() - starting_time
        logger.info(f'time {duration[iteration]} secs')

    return labeled_image, energy, duration


def calculate_energy(observed_image: np.ndarray, labeled_image: np.ndarray,
                     mask_image: np.ndarray, lambda_value: int,
                     max_smoothness_penalty: float) -> int:
    # E = Ep + Es
    #
    # Ep = Sum(Dp(lp))
    #
    # Dp(lp) = 0, if lp is damaged
    # Dp(lp)  = (lp-op)^2, else
    #
    # Es = lambda * Sum(min(Vmax, Vpq(lp,lq)))
    # Vpq(lp,lq) = (lp-lq)^2
    image_height, image_width = labeled_image.shape
    data_energy = 0
    smoothness_energy = 0
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

    smoothness_energy *= lambda_value

    energy = data_energy + smoothness_energy

    logger.info(f'Energy = {energy}, Data energy = {data_energy},'
                f'Smoothness energy = {smoothness_energy}')

    return energy


if __name__ == '__main__':
    main(sys.argv[1:])
