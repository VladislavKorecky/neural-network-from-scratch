import numpy as np

from nn.activations.activation_interface import ActivationFunctionInterface


class Layer:
    """
    Layer of connections in a neural network. Defines the connections/transformation between two layers and not the
    neurons themselves.

    Args:
        num_of_inputs (int): Number of inputs of the connections. In other words, number of neurons in the first layer.
        num_of_outputs (int): Number of outputs of the connections. In other words, number of neurons in the second
            layer.
    """

    def __init__(self, num_of_inputs: int, num_of_outputs: int, activation_function: ActivationFunctionInterface):
        self.num_of_inputs = num_of_inputs
        self.num_of_outputs = num_of_outputs

        self.activation_function = activation_function

        # initialize the weight and bias tensor with random values between -1 and 1.
        self.weight_matrix = np.random.uniform(-1, 1, (num_of_outputs, num_of_inputs))
        self.bias_vector = np.random.uniform(-1, 1, num_of_outputs)

        # variables used for backpropagation
        self.input = None  # last input of this layer
        self.linear_output = None  # last linear output of this layer

        self.weight_gradients = []  # list of calculated weight gradients
        self.bias_gradients = []  # list of calculated bias gradients

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Return the result/output of the layer's forward pass.

        Args:
            x (ndarray): Layer's input as a numpy array.

        Returns:
            ndarray: Layer's output as a numpy array.
        """

        # store the input as it will be useful later in backpropagation
        self.input = x

        # multiply each input by the weight and sum them up
        weighted_sum = np.dot(self.weight_matrix, x)

        # add the bias to the weighted sum
        linear_output = weighted_sum + self.bias_vector

        # store the linear output as it will be useful later in backpropagation
        self.linear_output = linear_output

        # apply the activation function
        return self.activation_function.apply(linear_output)

    def zero_grad(self) -> None:
        """
        Remove the stored gradients and other backpropagation variables.
        """

        self.input = None
        self.linear_output = None

        self.weight_gradients = []
        self.bias_gradients = []

    def backward(self, previous_gradient: np.ndarray) -> np.ndarray:
        """
        Use the information from previous forward passes to calculate the gradient for the weights and biases.

        Args:
            previous_gradient (ndarray): Stored gradient from the previous layers
                (previous in the context of backprop not forward) used to calculate the chain rule.

        Returns:
            ndarray: New gradient that will be sent to the next layer in backpropagation to continue the chain rule.
        """

        if self.input is None or self.linear_output is None:
            raise RuntimeError("No forward pass information found. Run forward() before trying to backpropagate the "
                               "network.")

        # calculate the derivative of the activation function with respect to it's input
        activation_gradient = self.activation_function.derive(self.linear_output)

        # calculate how far off was this layer's output from what it should have been
        delta = previous_gradient * activation_gradient

        # calculate the gradient (adjustment) of the weight
        weight_gradient = np.outer(delta, self.input)

        # calculate the gradient (adjustment) of the bias
        bias_gradient = delta

        # store the gradients
        self.weight_gradients.append(weight_gradient)
        self.bias_gradients.append(bias_gradient)

        # calculate the "previous_gradient" variable for the next step
        return np.dot(self.weight_matrix.T, delta)

    def update_parameters(self, learning_rate: float):
        """
        Update the weights and biases using the stored gradients.

        Args:
            learning_rate (float): Learning rate - determines the update size.
        """

        if self.weight_gradients == [] or self.bias_gradients == []:
            raise RuntimeError("No gradients found. Run backward() before trying to update the parameters.")

        # convert a list of tensors to one multi-dimensional tensor
        stacked_weight_gradients = np.stack(self.weight_gradients)
        stacked_bias_gradients = np.stack(self.bias_gradients)

        # average the gradients for multiple examples before applying it to the parameters
        self.weight_matrix -= learning_rate * stacked_weight_gradients.mean(axis=0)
        self.bias_vector -= learning_rate * stacked_bias_gradients.mean(axis=0)
