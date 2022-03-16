# --------------------------------------------------------------------------- #
# FILE:     FFMLP.py                                                          #
#                                                                             #
# PURPOSE:  create feedforward multilayer perceptron                          #
# --------------------------------------------------------------------------- #

# Planning:
# 28*28 grid input = 784 nodes in input layer.
# I will use 2 hidden layers, the first with 392 nodes and the second with 196 nodes.
# The output layer has 10 nodes; one for each classification of clothing.

from asyncio.windows_events import NULL
import numpy as np
import pandas as pd
from pyparsing import null_debug_action
from scipy.stats import truncnorm


input_size = 784
h1_size = 392
h2_size = 196
output_size = 10
learning_rate = 0.001


def read_data():
    return pd.read_csv('fashion_data/fashion-mnist_train.csv'), pd.read_csv('fashion_data/fashion-mnist_test.csv')


# Leaky ReLu Activation
def relu(z):
    return np.where(z > 0, z, 0.01 * z)


def relu_deriv(z):
    return np.where(z > 0, 1, 0.01)


# softmax
# def softmax(vector):
#     e = np.exp(vector)
#     return e / e.sum()


# def softmax_deriv(vector):
#     return np.diagflat(vector) - np.dot(vector, vector.T)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cat_cross_loss(output_vector, target_index):
    return -(np.log(output_vector[target_index]))


def cat_cross_loss_deriv(output_vector, target_index):
    return -1 / output_vector[target_index]


def initialise_weights(input_size, h1_size, h2_size, output_size):
    input_h1_weights = np.random.normal(1, 0.1, size = (h1_size, input_size))
    h1_h2_weights = np.random.normal(1, 0.1, size = (h2_size, h1_size))
    h2_output_weights = np.random.normal(1, 0.1, size = (output_size, h2_size))
    return(input_h1_weights, h1_h2_weights, h2_output_weights)


def initialise_biases(h1_size, h2_size, output_size):
    h1_biases = np.random.normal(1, 0.1, size = (h1_size, 1))
    h2_biases = np.random.normal(1, 0.1, size = (h2_size, 1))
    output_biases = np.random.normal(1, 0.1, size = (output_size, 1))
    return h1_biases, h2_biases, output_biases


def mnist_row_to_input(df, row):
    return np.array(df.iloc[row].values.tolist()[1:], ndmin=2).T, df.iloc[row][0]


# Runs a forward pass using input data (no classification), weights, and biases
def forward_propagate(input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    input_vector = input_with_target[0]
    h1_vector = relu(np.dot(input_h1_weights, input_vector) + h1_biases)
    h2_vector = relu(np.dot(h1_h2_weights, h1_vector) + h2_biases)
    output_vector = sigmoid(np.dot(h2_output_weights, h2_vector) + output_biases)
    return output_vector


# Returns classification, classified probability
def classify(output_vector):
    classification = np.where(output_vector == np.amax(output_vector))[0][0]
    classification_probability = np.amax(output_vector)
    return classification, classification_probability

### BACKPROPAGATION
# Resource 1: https://doug919.github.io/notes-on-backpropagation-with-cross-entropy/
# Resource 2: https://github.com/JohnPaton/numpy-neural-networks/blob/master/02-multi-layer-perceptron.ipynb


def backpropagate(input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases, learning_rate):
    input_vector = input_with_target[0]
    target_index = input_with_target[1]
    h1_z = np.dot(input_h1_weights, input_vector) + h1_biases
    h1_a = relu(h1_z)
    h2_z = np.dot(h1_h2_weights, h1_a) + h2_biases
    h2_a = relu(h2_z)
    output_z = np.dot(h2_output_weights, h2_a) + output_biases
    output_a = sigmoid(output_z)
    print(output_a)
    print(target_index)

    #print(cat_cross_loss_deriv(output_a, target_index))
    output_delta = cat_cross_loss_deriv(output_a, target_index) * sigmoid_deriv(output_z)
    h2_delta = np.matmul(output_delta, h2_output_weights) * relu_deriv(h2_z)
    h1_delta = np.matmul(h2_delta, h1_h2_weights) * relu_deriv(h1_z)

    h2_output_weights -= learning_rate * np.multiply(output_delta, h2_a)
    h1_h2_weights -= learning_rate * np.multiply(h2_delta, h1_a)
    input_h1_weights -= learning_rate * np.multiply(h1_delta, input_vector)

    output_biases -= learning_rate * output_delta
    h2_biases -= learning_rate * h2_delta
    h1_biases -= learning_rate * h1_delta

    return input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases


def main():
    df_train, df_test = read_data()

    input_h1_weights, h1_h2_weights, h2_output_weights = initialise_weights(input_size, h1_size, h2_size, output_size)
    h1_biases, h2_biases, output_biases = initialise_biases(h1_size, h2_size, output_size)
    input_with_target = mnist_row_to_input(df_train, 2)

    output_vector = forward_propagate(input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)

    #print(input_h1_weights)
    input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases = backpropagate(
        input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, 
        h1_biases, h2_biases, output_biases, learning_rate)
    #print(input_h1_weights)


if __name__ == '__main__':
    main()
