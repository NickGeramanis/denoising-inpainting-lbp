import numpy as np

from denoising_inpainting_lbp.node import Node

_N_LABELS = 256


def test_calculate_belief():
    node = Node(1, True)
    node.incoming_message_up = np.ones(_N_LABELS, dtype=np.int64)
    node.incoming_message_down = 2 * np.ones(_N_LABELS, dtype=np.int64)
    node.incoming_message_left = 3 * np.ones(_N_LABELS, dtype=np.int64)
    node.incoming_message_right = 4 * np.ones(_N_LABELS, dtype=np.int64)

    node.calculate_belief()

    assert np.array_equal(node.belief, 10 * np.ones(_N_LABELS, dtype=np.int64))


def test_calculate_label():
    node = Node(1, False)
    node.calculate_belief()

    node.calculate_label()

    assert node.label == 1
