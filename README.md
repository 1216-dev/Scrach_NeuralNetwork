# Scrach_NeuralNetwork
# Deep Neural Network from Scratch

This repository contains the implementation of a deep neural network (DNN) built from scratch using NumPy. The project includes forward propagation, backward propagation, and gradient descent optimization without using any high-level libraries like TensorFlow or PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Network Architecture](#network-architecture)
- [Activation Functions](#activation-functions)
- [Cost Function](#cost-function)
- [Forward and Backward Propagation](#forward-and-backward-propagation)
- [Training the Model](#training-the-model)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [License](#license)

## Introduction

This project aims to implement a multi-layer deep neural network from scratch, showcasing the understanding of core neural network principles without relying on high-level deep learning libraries like Keras or TensorFlow.

The network is trained on a dataset to perform binary classification, with the goal of achieving high accuracy on both training and test data.

## Dataset

The dataset used for this project is stored in CSV files:

- `cat_train_x.csv`: The training images (flattened 64x64x3 images)
- `cat_train_y.csv`: The corresponding labels for the training set (0 or 1 for binary classification)
- `cat_test_x.csv`: The test images
- `cat_test_y.csv`: The corresponding labels for the test set

The input features are normalized by dividing by 255 to scale pixel values between 0 and 1.

## Network Architecture

The deep neural network consists of L layers, with the following architecture:

1. **Input layer**: Number of neurons = 12,288 (64x64x3 flattened image)
2. **Hidden layers**: Fully connected layers with ReLU/Tanh activation
   - **Layer 1**: 100 neurons
   - **Layer 2**: 200 neurons
3. **Output layer**: 1 neuron for binary classification (sigmoid activation)

This model supports using either ReLU or Tanh as the activation function for the hidden layers, with the sigmoid function used in the output layer for binary classification.

## Activation Functions

The following activation functions are implemented:

- **ReLU**: Rectified Linear Unit, used in hidden layers.
- **Tanh**: Hyperbolic tangent activation function, an alternative to ReLU.
- **Sigmoid**: Used in the output layer for binary classification.
- **Softmax**: Used for multi-class classification if needed.

## Cost Function

The cost function for binary classification is computed using the cross-entropy loss:

\[
\text{Cost} = -\frac{1}{m} \sum_{i=1}^{m} \left[ Y^{(i)} \log(A^{(i)}) + \left(1 - Y^{(i)}\right) \log\left(1 - A^{(i)}\right) \right]
\]

Where \( m \) is the number of training examples.

## Forward and Backward Propagation

### Forward Propagation

- The data flows forward through the network, starting from the input layer through all hidden layers to the output layer.
- The activation function (ReLU or Tanh) is applied after each layer.
- In the final layer, the sigmoid function is applied for binary classification.

### Backward Propagation

- Gradients are computed using the chain rule to calculate the partial derivatives of the cost function with respect to the parameters.
- The parameters are updated using gradient descent.

## Training the Model

The training loop consists of the following steps:

1. **Initialize** the weights and biases.
2. **Forward propagation** to compute predictions.
3. **Calculate** the cost function.
4. **Backward propagation** to compute gradients.
5. **Update parameters** using gradient descent.
6. **Repeat** for a specified number of iterations.

Activation Functions
The following activation functions are implemented:

ReLU: Rectified Linear Unit, used in hidden layers.
Tanh: Hyperbolic tangent activation function, an alternative to ReLU.
Sigmoid: Used in the output layer for binary classification.
Softmax: Used for multi-class classification if needed.
## Activation Functions

The following activation functions are implemented:

- **ReLU**: Rectified Linear Unit, used in hidden layers.
- **Tanh**: Hyperbolic tangent activation function, an alternative to ReLU.
- **Sigmoid**: Used in the output layer for binary classification.
- **Softmax**: Used for multi-class classification if needed.

## Cost Function

The cost function for binary classification is computed using the cross-entropy loss:

\[
\text{Cost} = -\frac{1}{m} \sum_{i=1}^{m} \left[ Y^{(i)} \log(A^{(i)}) + \left(1 - Y^{(i)}\right) \log\left(1 - A^{(i)}\right) \right]
\]

Where:
- \( m \) is the number of training examples.
- \( Y \) is the actual label.
- \( A \) is the predicted output.

## Forward and Backward Propagation

### Forward Propagation

- Data flows forward through the network, starting from the input layer, passing through all hidden layers, and ending at the output layer.
- The activation function (ReLU or Tanh) is applied after each hidden layer.
- In the final layer, the sigmoid function is applied for binary classification.

### Backward Propagation

- Gradients are computed using the chain rule to calculate the partial derivatives of the cost function with respect to the parameters.
- The parameters are updated using gradient descent.

## Training the Model

The training loop consists of the following steps:

1. **Initialize** the weights and biases.
2. **Perform forward propagation** to compute predictions.
3. **Calculate** the cost function.
4. **Perform backward propagation** to compute gradients.
5. **Update parameters** using gradient descent.
6. **Repeat** for a specified number of iterations.

## Results

The model is trained for a fixed number of iterations, and the training/testing accuracy and cost function are logged at intervals.

Example training results:


## How to Run

1. **Clone** the repository:

   ```bash
   git clone https://github.com/yourusername/deep-neural-network-scratch.git


Make sure to adjust the GitHub repository URL (`yourusername`) to your actual username. This formatted text will render nicely on GitHub.



