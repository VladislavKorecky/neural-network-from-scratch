import numpy as np

from nn.loss_functions.loss_function_interface import LossFunctionInterface


class MeanSquaredError(LossFunctionInterface):
    """
    Implementation of Mean Squared Error (MSE) loss function. This function works by calculating the squared distance
    between the output and the target.
    """

    def distance(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.power(output - target, 2)

    def derive(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2 * (output - target)
