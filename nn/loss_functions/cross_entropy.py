import numpy as np

from nn.loss_functions.loss_function_interface import LossFunctionInterface


class CrossEntropy(LossFunctionInterface):
    """
    Implementation of Cross Entropy loss function. This function works only on softmax output.
    The target must be a one hot vector with 1 for the desired class.
    """

    def distance(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        # clip the values to avoid numerical instability
        clipped_output = np.clip(output, 1e-15, 1 - 1e-15)

        return -np.sum(target * np.log(clipped_output))

    def derive(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        # clip the values to avoid numerical instability
        clipped_output = np.clip(output, 1e-15, 1 - 1e-15)

        return clipped_output - target
