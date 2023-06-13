import numpy as np

from nn.network import NeuralNetwork


def test_network(net: NeuralNetwork, dataset: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate a network's accuracy on a given dataset.

    Args:
        net (NeuralNetwork): Neural network to test.
        dataset (ndarray): Dataset to test the network on.
        labels (ndarray): Correct class for each image.

    Returns:
        float: Percentage accuracy of the network.
    """

    # binary list of prediction results, 1 = guessed correctly, 0 = guessed incorrectly
    results = []

    for image, label in zip(dataset, labels):
        # reshape the 28 x 28 image to 784 so that the network can process it
        flat_image = image.reshape(784)

        # normalize the image to values between 0 and 1 instead of 0 and 255
        network_input = flat_image / np.linalg.norm(flat_image)

        # make a prediction with the network
        prediction = net.forward(network_input)
        net.zero_grad()

        # turn the network output (a vector) into a single number representing the chosen class
        class_prediction = np.argmax(prediction)

        # check if the prediction is correct
        results.append(class_prediction == label)

    # get the accuracy by taking the average of the results
    return sum(results) / len(results) * 100
