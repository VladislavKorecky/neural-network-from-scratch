import numpy as np

from abc import ABC, abstractmethod


class ActivationFunctionInterface(ABC):
    """
    Interface for defining activation functions.
    """

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply an activation function to a tensor.

        Args:
            x (ndarray): Numpy array (tensor) to which the activation function is applied.

        Returns:
            ndarray: Result after applying the activation function
        """

        pass

    @abstractmethod
    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Return the activation's derivative at certain point.

        Args:
            x (ndarray): Points on the x-axis where the derivative is calculated.

        Returns:
            ndarray: A derivative for each input in "x".
        """

        pass
