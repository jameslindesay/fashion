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
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def read_data():
    return pd.read_csv('fashion_data/fashion-mnist_train.csv'), pd.read_csv('fashion_data/fashion-mnist_test.csv')


# Leaky ReLu Activation
def relu(z):
    return np.where(z > 0, z, 0.01 * z)


def relu_deriv(z):
    return np.where(z > 0, 1, 0.01)


def softmax(vector):
    e = np.exp(vector - np.max(vector))
    return e / e.sum()


def cat_cross_loss(output_vector, target_index):
    return -(np.log(output_vector[target_index] + 1e-100))


def cat_cross_loss_deriv(output_vector, target_index):
    return -1 / output_vector[target_index]


def initialise_weights(input_size, h1_size, h2_size, output_size):
    input_h1_weights = np.random.randn(h1_size, input_size) * np.sqrt(1 / h1_size)
    h1_h2_weights = np.random.randn(h2_size, h1_size) * np.sqrt(1 / h2_size)
    h2_output_weights = np.random.randn(output_size, h2_size) * np.sqrt(1 / output_size)
    return input_h1_weights, h1_h2_weights, h2_output_weights


def initialise_biases(h1_size, h2_size, output_size):
    h1_biases = np.random.randn(h1_size, 1)
    h2_biases = np.random.randn(h2_size, 1)
    output_biases = np.random.randn(output_size, 1)
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


### BACKPROPAGATION
# Resource 1: https://doug919.github.io/notes-on-backpropagation-with-cross-entropy/
# Resource 2: https://github.com/JohnPaton/numpy-neural-networks/blob/master/02-multi-layer-perceptron.ipynb


def backpropagate(input_with_target, learning_rate, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    input_vector = input_with_target[0]
    target_index = input_with_target[1]
    target_vector = np.zeros((output_biases.shape[0], 1))
    target_vector[target_index] = 1

    h1_z = np.dot(input_h1_weights, input_vector) + h1_biases # 392 * 1
    h1_a = relu(h1_z) # 392 * 1
    h2_z = np.dot(h1_h2_weights, h1_a) + h2_biases # 196 * 1
    h2_a = relu(h2_z) # 196 * 1
    output_z = np.dot(h2_output_weights, h2_a) + output_biases # 10 * 1
    output_a = softmax(output_z) # 10 * 1

    output_delta = (output_a - target_vector).T # 1 * 10
    h2_delta = np.dot(output_delta, h2_output_weights) * relu_deriv(h2_z).T
    h1_delta = np.dot(h2_delta, h1_h2_weights) * relu_deriv(h1_z).T

    h2_output_weights -= learning_rate * np.multiply(output_delta.T, h2_a.T) # 10 * 196
    h1_h2_weights -= learning_rate * np.multiply(h2_delta.T, h1_a.T) # 196 * 392
    input_h1_weights -= learning_rate * np.multiply(h1_delta.T, input_vector.T) # 392 * 784

    output_biases -= learning_rate * output_delta.T
    h2_biases -= learning_rate * h2_delta.T
    h1_biases -= learning_rate * h1_delta.T

    return input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases


def train(train_size, df, epochs, learning_rate, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    losses = [0] * epochs
    accuracies = [0] * epochs
    for epoch in tqdm(range(epochs), desc = "Epochs"):
        loss = 0
        corrects = [0] * train_size
        indices = random.sample(range(0, len(df)), train_size)
        for index in range(len(indices)):
            input_with_target = mnist_row_to_input(df, indices[index])
            output_vector = forward_propagate(input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
            error = cat_cross_loss(output_vector, input_with_target[1])
            loss += error
            if np.argmax(output_vector) == input_with_target[1]:
                corrects[index] = 1
            else:
                corrects[index] = 0
            input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases = backpropagate(input_with_target, learning_rate, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
        losses[epoch] = loss
        accuracies[epoch] = sum(corrects)/len(corrects)
    return losses, accuracies, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases

def plot_loss_accu(epochs, losses, accuracies):
    ax = plt.gca()
    ax2 = plt.twinx()
    ax.plot(range(epochs), losses, 'b')
    ax2.plot(range(epochs), accuracies, 'r')
    ax.set_ylabel("Loss", fontsize = 14, color = 'r')
    ax2.set_ylabel("Accuracy", fontsize = 14, color = 'b')
    ax.set_xlabel("Epoch", fontsize = 14, color = 'black')
    plt.title("Loss & Accuracy by Epoch", fontsize = 20, color = 'black')
    plt.show()


def main():
    input_size = 784
    h1_size = 100
    h2_size = 100
    output_size = 10
    learning_rate = 1e-4
    
    print("Loading data...")
    df_train, df_test = read_data()
    print("Data loaded!")
    input_h1_weights, h1_h2_weights, h2_output_weights = initialise_weights(input_size, h1_size, h2_size, output_size)
    h1_biases, h2_biases, output_biases = initialise_biases(h1_size, h2_size, output_size)

    train_size = 10000
    epochs = 10
    losses, accuracies, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases = train(train_size, df_train, epochs, learning_rate, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
    
    # Plot loss and accuracy over epochs
    plot_loss_accu(epochs, losses, accuracies)

    input_with_target = mnist_row_to_input(df_test, 0)
    output_vector = forward_propagate(input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
    print(np.argmax(output_vector))
    print(input_with_target[1])

if __name__ == '__main__':
    main()
