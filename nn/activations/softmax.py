import numpy as np

from nn.activations.activation_interface import ActivationFunctionInterface


class Softmax(ActivationFunctionInterface):
    """
    Implementation of the softmax activation function.
    Softmax converts and array of real numbers into probabilities between 0 and 1 (0 = 0%, 1 = 100%).
    """

    def apply(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)

    def derive(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.apply(x)
        return probabilities * (1 - probabilities)
