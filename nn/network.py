import numpy as np

from nn.layer import Layer
from nn.loss_functions.loss_function_interface import LossFunctionInterface


class NeuralNetwork:
    """
    Simple neural network.

    Args:
        layers (list[Layer]): List of layers that the network uses.
        loss_function (LossFunctionInterface): Loss/Error/Cost function used for updating the network.
    """

    def __init__(self, layers: list[Layer], loss_function: LossFunctionInterface):
        self.layers = layers
        self.loss_function = loss_function

        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Return the result/output of a feed-forward pass.

        Args:
            x (ndarray): Neural network's input as a numpy array.

        Returns:
            ndarray: Neural network's output as a numpy array.
        """

        output = x

        for layer in self.layers:
            output = layer.forward(output)

        # store the output for later use in backpropagation
        self.output = output

        return output

    def zero_grad(self) -> None:
        """
        Remove the stored gradients and other backpropagation variables.
        """

        for layer in self.layers:
            layer.zero_grad()

        self.output = None

    def backward(self, target: np.ndarray) -> float:
        """
        Backpropagate the neural network using information from previous forward passes and the target.

        Args:
            target (ndarray): Correct/Desired network output. Sometimes referred to as the target or a label.

        Returns:
            float: Total loss/error/cost.
        """

        if self.output is None:
            raise RuntimeError("No forward pass information found. Run forward() before trying to backpropagate the "
                               "network.")

        # calculate the derivative of the loss function
        gradient = self.loss_function.derive(self.output, target)

        # loop from the end of the layers to the start (propagating backwards)
        for layer in reversed(self.layers):

            # use the calculated gradients from the previous layers in the next layers to calculate the chain rule
            gradient = layer.backward(gradient)

        # calculate the loss to serve as a measure of performance
        return self.loss_function.distance(self.output, target).mean()

    def update_parameters(self, learning_rate: float) -> None:
        """
        Update the weights and biases using the stored gradients.

        Args:
            learning_rate (float): Learning rate - determines the update size.
        """

        for layer in self.layers:
            layer.update_parameters(learning_rate)
