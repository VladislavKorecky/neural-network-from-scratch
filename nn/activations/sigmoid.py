import numpy as np

from nn.activations.activation_interface import ActivationFunctionInterface


class Sigmoid(ActivationFunctionInterface):
    """
    Implementation of the sigmoid activation function. Sigmoid non-linearly squishes any real number between 0 and 1.
    """

    def apply(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derive(self, x: np.ndarray) -> np.ndarray:
        activated_input = self.apply(x)
        return activated_input * (1 - activated_input)
