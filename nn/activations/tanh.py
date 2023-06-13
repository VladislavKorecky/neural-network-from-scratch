import numpy as np

from nn.activations.activation_interface import ActivationFunctionInterface


class Tanh(ActivationFunctionInterface):
    """
    Implementation of the tanh activation function. Tanh non-linearly squishes a real number between -1 and 1.
    """

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derive(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.power(np.tanh(x), 2)
