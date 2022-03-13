# --------------------------------------------------------------------------- #
# FILE:     FFMLP.py                                                          #
#                                                                             #
# PURPOSE:  create feedforward multilayer perceptron                          #
# --------------------------------------------------------------------------- #

# Planning:
# 28*28 grid input = 784 nodes in input layer.
# I will use 2 hidden layers, the first with 392 nodes and the second with 196 nodes.
# The output layer has 10 nodes; one for each classification of clothing.

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

input_size = 784
h1_size = 392
h2_size = 196
output_size = 10
learning_rate = 0.01

def read_data():
    return pd.read_csv('fashion_data/fashion-mnist_train.csv'), pd.read_csv('fashion_data/fashion-mnist_test.csv')

# Used in setting initial values for weights
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# Sigmoid activation
# def activate(x):
#     return 1/(1 + np.exp(-x))

# softmax
def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()

# Leaky ReLu Activation
def ReLU(x):
    if x > 0:
        return x
    else:
        return 0.01 * x
# vectorise for use with arrays
ReLU_v = np.vectorize(ReLU)

def ReLU_deriv(x):
    if x > 0:
        return 1
    else:
        return 0.01
# vectorise for use with arrays
ReLU_deriv_v = np.vectorize(ReLU_deriv)

def initialise_weights(input_size, h1_size, h2_size, output_size):
    input_h1_weights = truncated_normal(mean=0, sd=1, low=-1/np.sqrt(input_size), upp=1/np.sqrt(input_size)).rvs((h1_size, input_size))
    h1_h2_weights = truncated_normal(mean=0, sd=1, low=-1/np.sqrt(h1_size), upp=1/np.sqrt(h1_size)).rvs((h2_size, h1_size))
    h2_output_weights = truncated_normal(mean=0, sd=1, low=-1/np.sqrt(h2_size), upp=1/np.sqrt(h2_size)).rvs((output_size, h2_size))
    return(input_h1_weights, h1_h2_weights, h2_output_weights)

def initialise_biases(h1_size, h2_size, output_size):
    h1_biases = np.array([0.5]*h1_size, ndmin=2).T
    h2_biases = np.array([0.5]*h2_size, ndmin=2).T
    output_biases = np.array([0.5]*output_size, ndmin=2).T
    return h1_biases, h2_biases, output_biases

def mnist_row_to_input_vector(df, row):
    return np.array(df.iloc[row].values.tolist()[1:], ndmin=2).T

def forward_propagate(input_data, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    input_vector = input_data
    h1_vector = ReLU_v(np.matmul(input_h1_weights, input_vector) + h1_biases)
    h2_vector = ReLU_v(np.matmul(h1_h2_weights, h1_vector) + h2_biases)
    output_vector = softmax(np.matmul(h2_output_weights, h2_vector) + output_biases)
    return input_vector, h1_vector, h2_vector, output_vector

# Returns classification, classified probability, target probability
def classify(input_data, df, input_index, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    a, b, c, output_vector = forward_propagate(input_data, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
    return np.where(output_vector == np.amax(output_vector))[0][0], np.amax(output_vector), output_vector[df.iloc[input_index]['label']][0]

def classification_loss(target_probability):
    return -(np.log(target_probability))

def forward_propagate_batch(input_indices, df, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    classifications = []
    for index in input_indices:
        input_data = mnist_row_to_input_vector(df, index)
        classification, classification_probability, target_probability = classify(input_data, df, index, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
        error = classification_loss(target_probability)
        classifications.append((index, classification, df.iloc[index]['label'], classification_probability, target_probability, error))    
    return classifications

# def calculate_batch_loss(classifications):
#     loss = 0
#     for classification in classifications:
#         loss -= (classification[1] == classification[2]) * np.log(classification[3])
#     return loss

### BACKPROPAGATION: https://doug919.github.io/notes-on-backpropagation-with-cross-entropy/

# def backpropagate_batch(input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases, learning_rate):
#     input_h1_weights_updated = -learning_rate * 
#     return input_h1_weights_updated, h1_h2_weights_updated, h2_output_weights_updated, h1_biases_updated, h2_biases_updated, output_biases_updated

# TODO: backpropagate, run on test

def main():
    df_train, df_test = read_data()
    h1_biases, h2_biases, output_biases = initialise_biases(h1_size, h2_size, output_size)
    input_h1_weights, h1_h2_weights, h2_output_weights = initialise_weights(input_size, h1_size, h2_size, output_size)
    classifications = forward_propagate_batch([i for i in range(10)], df_train, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
    print(classifications)

if __name__ == '__main__':
    main()
