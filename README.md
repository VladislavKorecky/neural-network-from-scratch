# Neural Network from Scratch

A modular neural network implementation in Python using NumPy. This project implements basic deep learning components without using high-level frameworks like PyTorch or TensorFlow.

The library includes a demonstration of training a classifier on the MNIST dataset.

## Features

- **Modular Architecture**: Define networks with arbitrary layers and neurons.
- **Supported Components**:
  - **Activation Functions**: ReLU, Leaky ReLU, Sigmoid, Softmax, Tanh, and Linear.
  - **Loss Functions**: Mean Squared Error (MSE) and Cross-Entropy.
- **Backpropagation**: Implementation of the backpropagation algorithm for gradient calculation.
- **Batch Training**: Support for mini-batch gradient descent.
- **MNIST Integration**: Utilities for loading the MNIST dataset from `.ubyte` files.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- idx2numpy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

To train the network on the MNIST dataset:

```bash
python main.py
```

The script will:
1. Load MNIST data from the `mnist/` directory.
2. Initialize a 3-layer network.
3. Train the network and plot accuracy over time.
4. Output final training and testing accuracy.

## Architecture

- `nn/network.py`: `NeuralNetwork` class for forward and backward passes.
- `nn/layer.py`: `Layer` class for weight and bias management.
- `nn/activations/`: Implementation of activation functions.
- `nn/loss_functions/`: Implementation of loss functions.
- `utils.py`: Testing and accuracy utilities.

## Example Usage

```python
from nn.network import NeuralNetwork
from nn.layer import Layer
from nn.activations.tanh import Tanh
from nn.activations.softmax import Softmax
from nn.loss_functions.cross_entropy import CrossEntropy

# Initialize network
net = NeuralNetwork([
    Layer(784, 16, Tanh()),
    Layer(16, 16, Tanh()),
    Layer(16, 10, Softmax())
], CrossEntropy())

# Training step
output = net.forward(input_data)
loss = net.backward(target_label)
net.update_parameters(learning_rate=0.1)
net.zero_grad()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
