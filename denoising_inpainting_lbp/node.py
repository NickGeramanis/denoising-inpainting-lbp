"""This module contains the Node class."""
import numpy as np


class Node:
    """A node of a Markov random field."""
    _N_LABELS = 256

    incoming_message_left: np.ndarray
    incoming_message_right: np.ndarray
    incoming_message_up: np.ndarray
    incoming_message_down: np.ndarray
    data_cost: np.ndarray
    belief: np.ndarray
    label: int

    def __init__(self, label: int, zero_data_cost: bool) -> None:
        self.incoming_message_left = np.zeros(self._N_LABELS, dtype=np.int64)
        self.incoming_message_right = np.zeros(self._N_LABELS, dtype=np.int64)
        self.incoming_message_up = np.zeros(self._N_LABELS, dtype=np.int64)
        self.incoming_message_down = np.zeros(self._N_LABELS, dtype=np.int64)
        self.data_cost = (np.zeros(self._N_LABELS) if zero_data_cost
                          else (label - np.arange(0, self._N_LABELS)) ** 2)
        self.label = label

    def calculate_belief(self) -> None:
        """Calculate the belief (probability) of the node."""
        self.belief = (self.data_cost
                       + self.incoming_message_left
                       + self.incoming_message_right
                       + self.incoming_message_up
                       + self.incoming_message_down)

    def calculate_label(self) -> None:
        """Calculate the minimum a posteriori probability (MAP) of the node."""
        self.label = int(np.argmin(self.belief))
