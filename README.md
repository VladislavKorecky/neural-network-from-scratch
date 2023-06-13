# MNIST Neural Network Classifier From Scratch
This repository contains code for a neural network classifier done only using the NumPy library (plus Matplotlib for visualization and idx2numpy to load the dataset). The neural network successfully learns to classify images from the famous MNIST dataset. The code also contains two loss functions (MSE and Cross Entropy), five activation functions (ReLU, LeakyReLU, Sigmoid, Softmax, Tanh) and supports batch/mini-batch gradient descent.

## Setup
In order to use this repository, you first need to install the required libraries. Move ("cd") to the project folder and run the following command: `pip install -r requirements.txt`

## Running the code
If you want to train the neural network you can run `main.py`. You'll see the network's accuracy printed in the terminal as the AI trains. Once the training is done, a graph will appear showing the accuracy over time. After the graph is closed, an automatic test will occur that calculates the network's accuracy for all data points in both the training and testing dataset.

## Config
You can configure the training process at the top of the `main.py` file. You can easily specify the number of epochs, the batch size, and the learning rate. If you scroll down to the setup section, you can modify the array of layers to include different activation functions, add/remove neurons, etc.