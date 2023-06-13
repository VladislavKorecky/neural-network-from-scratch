import numpy as np

from nn.activations.activation_interface import ActivationFunctionInterface


class ReLU(ActivationFunctionInterface):
    """
    Implementation of the ReLU activation function. ReLU is a simple and effective activation that returns the
    original number if it's greater than 0 and returns 0 otherwise.
    """

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derive(self, x: np.ndarray) -> np.ndarray:
        return 1. * (x > 0)
