import numpy as np

import matplotlib.pyplot as plt

from nn.activations.sigmoid import Sigmoid
from nn.activations.tanh import Tanh
from nn.layer import Layer
from nn.loss_functions.mse import MeanSquaredError
from nn.network import NeuralNetwork


# --------------
#     CONFIG
# --------------
EPOCHS = 10000
LEARNING_RATE = 0.1


# -------------
#     SETUP
# -------------
dataset = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

labels = np.array([
    [0],
    [1],
    [1],
    [0]
])

net = NeuralNetwork([
    Layer(2, 2, Tanh()),
    Layer(2, 1, Sigmoid())
], MeanSquaredError())

# keeps track of previous losses
loss_history = []


# --------------
#    TRAINING
# --------------
for epoch in range(EPOCHS):

    # tracks losses per epoch, in other words tracks the loss of each datapoint
    epoch_losses = []

    # go through each datapoint/label pair
    for datapoint, label in zip(dataset, labels):

        # make a prediction
        net.forward(datapoint)

        # calculate the loss and the gradients
        loss = net.backward(label)
        epoch_losses.append(loss)

    # get the average loss per datapoint and append it to the loss history
    loss = sum(epoch_losses) / len(epoch_losses)
    loss_history.append(loss)

    print(loss)

    # update the network's parameters using the gradients
    net.update_parameters(LEARNING_RATE)

    # delete the stored gradients
    net.zero_grad()


# plot the loss over time
plt.plot(loss_history)
plt.show()
