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
learning_rate = 0.01

def read_data():
    return pd.read_csv('fashion_data/fashion-mnist_train.csv'), pd.read_csv('fashion_data/fashion-mnist_test.csv')

# Leaky ReLu Activation
def relu(z):
    np.where(z > 0, z, 0.01 * z)

def relu_deriv(z):
    np.where(z > 0, 1, 0.01)

# softmax
def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()

def softmax_deriv(vector):
    # TODO
    return NULL

def initialise_weights(input_size, h1_size, h2_size, output_size):
    input_h1_weights = np.random.normal(0, 0.1, size = (h1_size, input_size))
    h1_h2_weights = np.random.normal(0, 0.1, size = (h2_size, h1_size))
    h2_output_weights = np.random.normal(0, 0.1, size = (output_size, h2_size))
    return(input_h1_weights, h1_h2_weights, h2_output_weights)

def initialise_biases(h1_size, h2_size, output_size):
    h1_biases = np.random.normal(0, 0.1, size = (h1_size, 1))
    h2_biases = np.random.normal(0, 0.1, size = (h2_size, 1))
    output_biases = np.random.normal(0, 0.1, size = (output_size, 1))
    return h1_biases, h2_biases, output_biases

def mnist_row_to_input(df, row):
    return np.array(df.iloc[row].values.tolist()[1:], ndmin=2).T, df.iloc[row][0]

# Runs a forward pass using input data (no classification), weights, and biases
def forward_propagate(input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    input_vector = input_with_target[0]
    h1_vector = relu(np.dot(input_h1_weights, input_vector) + h1_biases)
    h2_vector = relu(np.dot(h1_h2_weights, h1_vector) + h2_biases)
    output_vector = softmax(np.dot(h2_output_weights, h2_vector) + output_biases)
    return output_vector

# Returns classification, classified probability
def classify(output_vector):
    classification = np.where(output_vector == np.amax(output_vector))[0][0]
    classification_probability = np.amax(output_vector)
    return classification, classification_probability

def classification_loss(target_probability):
    return -(np.log(target_probability))

def forward_propagate_batch(input_indices, df, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    classifications = []
    for index in input_indices:
        input_with_target = mnist_row_to_input(df, index)
        classification, classification_probability, target_probability = classify(input_with_target, df, index, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
        error = classification_loss(target_probability)
        classifications.append((index, classification, df.iloc[index]['label'], classification_probability, target_probability, error))    
    return classifications

# def calculate_batch_loss(classifications):
#     loss = 0
#     for classification in classifications:
#         loss -= (classification[1] == classification[2]) * np.log(classification[3])
#     return loss

### BACKPROPAGATION: https://doug919.github.io/notes-on-backpropagation-with-cross-entropy/
### ALSO: https://github.com/JohnPaton/numpy-neural-networks/blob/master/02-multi-layer-perceptron.ipynb

# def backpropagate_batch(input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases, learning_rate):
#     input_h1_weights_updated = -learning_rate * 
#     return input_h1_weights_updated, h1_h2_weights_updated, h2_output_weights_updated, h1_biases_updated, h2_biases_updated, output_biases_updated

# TODO: backpropagate, run on test

def main():
    df_train, df_test = read_data()
    h1_biases, h2_biases, output_biases = initialise_biases(h1_size, h2_size, output_size)
    input_h1_weights, h1_h2_weights, h2_output_weights = initialise_weights(input_size, h1_size, h2_size, output_size)
    print(mnist_row_to_input(df_train, 2)[1])

if __name__ == '__main__':
    main()
