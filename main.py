import numpy as np

import idx2numpy

import matplotlib.pyplot as plt

from nn.activations.tanh import Tanh
from nn.activations.softmax import Softmax
from nn.layer import Layer
from nn.loss_functions.cross_entropy import CrossEntropy
from nn.network import NeuralNetwork
from utils import test_network

# --------------
#     CONFIG
# --------------
EPOCHS = 2000
BATCH_SIZE = 64
LEARNING_RATE = 1


# -------------
#     SETUP
# -------------

# load the MNIST dataset
training_dataset = idx2numpy.convert_from_file("mnist/train-images.idx3-ubyte")
testing_dataset = idx2numpy.convert_from_file("mnist/t10k-images.idx3-ubyte")

# load the MNIST labels
training_labels = idx2numpy.convert_from_file("mnist/train-labels.idx1-ubyte")
testing_labels = idx2numpy.convert_from_file("mnist/t10k-labels.idx1-ubyte")

# initialize the network
net = NeuralNetwork([
    Layer(784, 16, Tanh()),
    Layer(16, 16, Tanh()),
    Layer(16, 10, Softmax())
], CrossEntropy())

# keeps track of accuracy over time
accuracy_history = []


# --------------
#    TRAINING
# --------------
for epoch in range(EPOCHS):

    # get random indexes from the training data to form a batch
    batch_indexes = np.random.choice(training_dataset.shape[0], BATCH_SIZE, replace=False)

    # use the batch indexes to get the corresponding images and labels
    dataset_batch = training_dataset[batch_indexes]
    label_batch = training_labels[batch_indexes]

    # keep track of predicted classes
    predicted_classes = []

    # go through each sample in the batch
    for image, label in zip(dataset_batch, label_batch):
        # reshape the 28 x 28 image to 784 so that the network can process it
        flat_image = image.reshape(784)

        # normalize the image to values between 0 and 1 instead of 0 and 255
        network_input = flat_image / np.linalg.norm(flat_image)

        # convert the label into a one-hot vector
        one_hot_label = np.zeros(10)
        one_hot_label[label] = 1

        # make a prediction with the network
        prediction = net.forward(network_input)

        # turn the network output (a vector) into a single number representing the chosen class
        class_prediction = np.argmax(prediction)
        predicted_classes.append(class_prediction)

        # calculate the loss and the gradient using backpropagation
        loss = net.backward(one_hot_label)

    # get the difference between the labels and the predicted classes
    diff = label_batch != np.array(predicted_classes)

    # calculate the number of differences
    diff_count = diff.sum()

    # use the diff count to calculate accuracy
    accuracy = 100 - (diff_count / BATCH_SIZE * 100)
    accuracy_history.append(accuracy)

    print(accuracy)

    # update the network's parameters based on the gradients
    net.update_parameters(LEARNING_RATE)

    # delete the stored gradients
    net.zero_grad()

# plot the accuracy
plt.plot(accuracy_history)
plt.show()


# -------------
#    TESTING
# -------------

# calculate the percentage accuracy on both the training and testing dataset
training_accuracy = test_network(net, training_dataset, training_labels)
testing_accuracy = test_network(net, testing_dataset, testing_labels)

# print testing info
print("-------------------")
print(f"Training accuracy: {training_accuracy}%")
print(f"Testing accuracy: {testing_accuracy}%")
