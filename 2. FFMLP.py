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

def read_data():
    return pd.read_csv('fashion_data/fashion-mnist_train.csv'), pd.read_csv('fashion_data/fashion-mnist_test.csv')

# Used in setting initial values for weights
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# Sigmoid activation
def activate(x):
    return 1/(1 + np.exp(-x))

# Leaky ReLu Activation
# def activate(x):
#     if x>0 :
#         return x
#     else :
#         return 0.01*x

def initialise_weights(input_size, h1_size, h2_size, output_size):
    input_h1_weights = truncated_normal(mean=2, sd=1, low=-1/np.sqrt(input_size), upp=1/np.sqrt(input_size)).rvs((h1_size, input_size))
    h1_h2_weights = truncated_normal(mean=2, sd=1, low=-1/np.sqrt(h1_size), upp=1/np.sqrt(h1_size)).rvs((h2_size, h1_size))
    h2_output_weights = truncated_normal(mean=2, sd=1, low=-1/np.sqrt(h2_size), upp=1/np.sqrt(h2_size)).rvs((output_size, h2_size))
    return(input_h1_weights, h1_h2_weights, h2_output_weights)

def initialise_biases(h1_size, h2_size, output_size):
    h1_biases = np.array([0.5]*h1_size, ndmin=2).T
    h2_biases = np.array([0.5]*h2_size, ndmin=2).T
    output_biases = np.array([0.5]*output_size, ndmin=2).T
    return h1_biases, h2_biases, output_biases

def mnist_row_to_input_vector(df, row):
    return np.array(df.iloc[row].values.tolist()[1:], ndmin=2).T

def forward_propagate(input_data, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    input_vector = activate(input_data)
    h1_vector = activate(np.matmul(input_h1_weights, input_vector) + h1_biases)
    h2_vector = activate(np.matmul(h1_h2_weights, h1_vector) + h2_biases)
    output_vector = activate(np.matmul(h2_output_weights, h2_vector) + output_biases)
    return input_vector, h1_vector, h2_vector, output_vector

# TODO: run batch of training set, calculate loss, backpropagate, run on test

def main():
    df_train, df_test = read_data()
    h1_biases, h2_biases, output_biases = initialise_biases(h1_size, h2_size, output_size)
    input_h1_weights, h1_h2_weights, h2_output_weights = initialise_weights(input_size, h1_size, h2_size, output_size)
    input_data = mnist_row_to_input_vector(df_train, 0)
    input_vector, h1_vector, h2_vector, output_vector = forward_propagate(input_data, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
    print(input_vector.shape)
    print(output_vector)

if __name__ == '__main__':
    main()
