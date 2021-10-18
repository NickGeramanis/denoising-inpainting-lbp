"""This module contains the MarkovRandomField class."""
import numpy as np

from denoising_inpainting_lbp.node import Node


class MarkovRandomField:
    """A representation of a Markov random field (MRF) of an image."""
    _N_LABELS = 256

    _starting_labels: np.ndarray
    _mask: np.ndarray
    _lambda: int
    _max_smoothness_penalty: int
    _graph: np.ndarray
    _smoothness_cost: np.ndarray

    def __init__(self,
                 starting_labels: np.ndarray,
                 mask: np.ndarray,
                 lambda_: int,
                 max_smoothness_penalty: int) -> None:
        self._starting_labels = starting_labels
        self._mask = mask
        self._lambda = lambda_
        self._max_smoothness_penalty = max_smoothness_penalty

        height, width = starting_labels.shape
        self._graph = np.empty(starting_labels.shape, dtype=Node)
        for row in range(height):
            for column in range(width):
                self._graph[row, column] = Node(self._N_LABELS,
                                                starting_labels[row, column])

        self._smoothness_cost = np.empty((self._N_LABELS, self._N_LABELS),
                                         dtype=np.int64)

        for i in range(self._N_LABELS):
            for j in range(self._N_LABELS):
                self._smoothness_cost[i, j] = (
                        lambda_ * min(max_smoothness_penalty, (i - j) ** 2))

    def calculate_energy(self) -> int:
        """Calculate the energy of an image.

        E = Ep + Es

        Ep = Sum(Dp(lp))
        Dp(lp) = 0,         if lp is damaged
        Dp(lp) = (lp-op)^2, else

        Es = lambda * Sum(min(Vmax, Vpq(lp,lq)))
        Vpq(lp,lq) = (lp-lq)^2
        """
        height, width = self._graph.shape
        data_energy = 0
        smoothness_energy = 0
        for row in range(height):
            for column in range(width):
                if self._mask[row, column] > 0:
                    data_energy += (self._starting_labels[row, column]
                                    - self._graph[row, column].label) ** 2

                if row < height - 1:
                    label_difference = (
                            (self._graph[row, column].label
                             - self._graph[row + 1, column].label) ** 2)
                    smoothness_energy += min(self._max_smoothness_penalty,
                                             label_difference)
                if column < width - 1:
                    label_difference = (
                            (self._graph[row, column].label
                             - self._graph[row, column + 1].label) ** 2)
                    smoothness_energy += min(self._max_smoothness_penalty,
                                             label_difference)

        smoothness_energy *= self._lambda
        energy = data_energy + smoothness_energy
        return energy

    def calculate_messages(self) -> None:
        """Calculate the messages for each node using the min-sum algorithm."""
        height, width = self._graph.shape
        # Pass messages along rows.
        for row in range(height):
            # Forward pass (left to right).
            for column in range(width - 1):
                for label in range(self._N_LABELS):
                    self._graph[row, column + 1].incoming_message_left[
                        label] = (
                        np.amin(
                            self._graph[row, column].data_cost
                            + self._smoothness_cost[label, :]
                            + self._graph[row, column].incoming_message_up
                            + self._graph[row, column].incoming_message_down
                            + self._graph[row, column].incoming_message_left))

            # Backward pass (right to left).
            for column in range(width - 1, 0, -1):
                for label in range(self._N_LABELS):
                    self._graph[row, column - 1].incoming_message_right[
                        label] = (
                        np.amin(
                            self._graph[row, column].data_cost
                            + self._smoothness_cost[label, :]
                            + self._graph[row, column].incoming_message_up
                            + self._graph[row, column].incoming_message_down
                            + self._graph[row, column].incoming_message_right))

        # Pass messages along columns.
        for column in range(width):
            # Forward pass (top down).
            for row in range(height - 1):
                for label in range(self._N_LABELS):
                    self._graph[row + 1, column].incoming_message_up[label] = (
                        np.amin(
                            self._graph[row, column].data_cost
                            + self._smoothness_cost[label, :]
                            + self._graph[row, column].incoming_message_left
                            + self._graph[row, column].incoming_message_right
                            + self._graph[row, column].incoming_message_up))

            # Backward pass (bottom up).
            for row in range(height - 1, 0, -1):
                for label in range(self._N_LABELS):
                    self._graph[row - 1, column].incoming_message_down[
                        label] = (
                        np.amin(
                            self._graph[row, column].data_cost
                            + self._smoothness_cost[label, :]
                            + self._graph[row, column].incoming_message_left
                            + self._graph[row, column].incoming_message_right
                            + self._graph[row, column].incoming_message_down))

    def recover_map(self) -> None:
        """Recover the minimum a posteriori probability (MAP) for each node."""
        height, width = self._graph.shape
        for row in range(height):
            for column in range(width):
                self._graph[row, column].calculate_belief()
                self._graph[row, column].calculate_label()

    def retrieve_labels(self) -> np.ndarray:
        """Retrieve the labels from each node."""
        height, width = self._graph.shape
        labels = np.empty(self._graph.shape, dtype=np.int64)
        for row in range(height):
            for column in range(width):
                labels[row, column] = self._graph[row, column].label

        return labels
