import numpy as np

from nn.activations.activation_interface import ActivationFunctionInterface


class LeakyReLU(ActivationFunctionInterface):
    """
    Implementation of the Leaky ReLU activation function.
    Leaky ReLU improves the original ReLU function by adding a slope to the negative part.
    """

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, x * 0.01)

    def derive(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0.01)
