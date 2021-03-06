# --------------------------------------------------------------------------- #
# FILE:     FFMLP.py                                                          #
#                                                                             #
# PURPOSE:  create feedforward multilayer perceptron                          #
# --------------------------------------------------------------------------- #

# Planning:
# 28*28 grid input = 784 nodes in input layer.
# I will use 2 hidden layers, the first with many nodes and the second with fewer nodes.
# The output layer has 10 nodes; one for each classification of clothing.


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_data():
    return pd.read_csv('fashion_data/fashion-mnist_train.csv'), pd.read_csv('fashion_data/fashion-mnist_test.csv')


# Leaky ReLU activation
def relu(z):
    return np.where(z > 0, z, 0.01 * z)


# Derivative of leaky ReLU activation
def relu_deriv(z):
    return np.where(z > 0, 1, 0.01)


# Softmax for output layer
def softmax(vector):
    e = np.exp(vector - np.max(vector))
    return e / e.sum()


# Categorical cross entropy loss function
def cat_cross_loss(output_vector, target_index):
    return -(np.log(output_vector[target_index] + 1e-100))


# Derivative of categorical cross entropy loss function
def cat_cross_loss_deriv(output_vector, target_index):
    return -1 / output_vector[target_index]


# Initialise NN weights to random standard normal values scaled by the size of the layers
# Scaling source: https://mlfromscratch.com/neural-network-tutorial/
def initialise_weights(input_size, h1_size, h2_size, output_size):
    input_h1_weights = np.random.randn(h1_size, input_size) * np.sqrt(1 / h1_size)
    h1_h2_weights = np.random.randn(h2_size, h1_size) * np.sqrt(1 / h2_size)
    h2_output_weights = np.random.randn(output_size, h2_size) * np.sqrt(1 / output_size)
    return input_h1_weights, h1_h2_weights, h2_output_weights


# Initialise NN biases to random standard normal values
def initialise_biases(h1_size, h2_size, output_size):
    h1_biases = np.random.randn(h1_size, 1)
    h2_biases = np.random.randn(h2_size, 1)
    output_biases = np.random.randn(output_size, 1)
    return h1_biases, h2_biases, output_biases


# Convert a pd df to a numpy array of the data with its target
def mnist_row_to_input(df, row):
    return (np.array(df.iloc[row].values.tolist()[1:], ndmin=2).T, df.iloc[row][0])


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


# Runs backpropagation
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


# Trains the network for a defined number of epochs and with batches of a defined size
def train(train_size, df, epochs, learning_rate, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    
    # TODO for each epoch it should train on full dataset, within each epoch it can train on subset of data...
    # TODO backpropagate once per batch rather than once per index (using average error)
    # TODO change loss to average rather than sum, then include loss in test function

    losses = [0] * epochs
    accuracies = [0] * epochs
    # for epoch in tqdm(range(epochs), desc = "Epoch"):
    for epoch in range(epochs):
        loss = 0
        corrects = [0] * train_size
        indices = random.sample(range(0, len(df)), train_size)
        for index in tqdm(range(len(indices)), desc = "Batch"):
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


# Runs a test pass on the network (similar to the training function)
def test(df, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    corrects = [0] * len(df)
    for index in range(len(df)):
        input_with_target = mnist_row_to_input(df, index)
        output_vector = forward_propagate(input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
        if np.argmax(output_vector) == input_with_target[1]:
            corrects[index] = 1
        else:
            corrects[index] = 0
    accuracy = sum(corrects)/len(corrects)
    return accuracy


# Plot the loss and accuracy over epochs on the same graph
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
    input_size = 784 # DO NOT CHANGE
    h1_size = 1000
    h2_size = 100
    output_size = 10 # DO NOT CHANGE
    learning_rate = 1e-5
    train_size = 5000
    epochs = 10
    
    print("Loading data...")
    df_train, df_test = read_data()
    print(max(mnist_row_to_input(df_train, 1)[0]))
    print("Data loaded!\n")
    input_h1_weights, h1_h2_weights, h2_output_weights = initialise_weights(input_size, h1_size, h2_size, output_size)
    h1_biases, h2_biases, output_biases = initialise_biases(h1_size, h2_size, output_size)

    print("Training...")
    losses, accuracies, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases = train(train_size, df_train, epochs, learning_rate, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)
    
    # Plot loss and accuracy over epochs
    plot_loss_accu(epochs, losses, accuracies)
    print("\nFinal training epoch loss: " + str(losses[-1][0]))
    print("Final training epoch accuracy: " + str(accuracies[-1]))

    print("\nRunning test...")
    accuracy = test(df_test, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)

    print("Test set accuracy: " + str(accuracy))

    print("\nSaving weights/biases")
    np.savetxt("out/input_h1_weights.csv", input_h1_weights, delimiter = ',')
    np.savetxt("out/h1_h2_weights.csv", h1_h2_weights, delimiter = ',')
    np.savetxt("out/h2_output_weights.csv", h2_output_weights, delimiter = ',')
    np.savetxt("out/h1_biases.csv", h1_biases, delimiter = ',')
    np.savetxt("out/h2_biases.csv", h2_biases, delimiter = ',')
    np.savetxt("out/output_biases.csv", output_biases, delimiter = ',')


if __name__ == '__main__':
    main()
