import numpy as np

from nn.activations.activation_interface import ActivationFunctionInterface


class Linear(ActivationFunctionInterface):
    """
    Implementation of a linear activation function. This activation simply returns it's input and thus has a
    derivative of 1.
    """

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x

    def derive(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape)
