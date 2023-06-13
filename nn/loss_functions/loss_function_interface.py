import numpy as np

from abc import ABC, abstractmethod


class LossFunctionInterface(ABC):
    @abstractmethod
    def distance(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Calculate the loss based on the output and the desired output.

        Args:
            output (ndarray): Network's prediction.
            target (ndarray): Correct/Desired network output. Sometimes referred to as the target of a label.

        Returns:
            ndarray: The loss (also referred to as error or cost) between the output and the target for each element.
        """

        pass

    @abstractmethod
    def derive(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the loss function with respect to the output.

        Args:
            output (ndarray): Network's prediction.
            target (ndarray): Correct/Desired network output. Sometimes referred to as the target of a label.

        Returns:
            ndarray: The derivative for each element in the output array.
        """

        pass
