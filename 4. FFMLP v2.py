# --------------------------------------------------------------------------- #
# FILE:     FFMLP.py                                                          #
#                                                                             #
# PURPOSE:  create feedforward multilayer perceptron                          #
# UPGRADE:  fixes batches and implemntation of training                       #
# --------------------------------------------------------------------------- #


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List


def read_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv("fashion_data/fashion-mnist_train.csv")
    test = pd.read_csv("fashion_data/fashion-mnist_test.csv")
    return train, test


# Leaky ReLU activation
def relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, 0.01 * z)


# Derivative of leaky ReLU activation
def relu_deriv(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0.01)


# Softmax for output layer
def softmax(matrix: np.ndarray) -> np.ndarray:
    col_max = np.max(matrix, axis=0, keepdims=True)
    e = np.exp(matrix - col_max)
    col_sum = np.sum(e, axis=0, keepdims=True)
    return e / col_sum


# Categorical cross entropy loss function
def cat_cross_loss(output_matrix: np.ndarray, targets: np.ndarray) -> float:
    mean_loss = np.mean(-np.log(output_matrix[targets[0], [range(len(targets[0]))]] + 1e-100))
    return mean_loss


# Initialise NN weights to random standard normal values scaled by the size of the layers
# Scaling source: https://mlfromscratch.com/neural-network-tutorial/
def initialise_weights(input_size: int, h1_size: int, h2_size: int, output_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_h1_weights = np.random.randn(h1_size, input_size) * np.sqrt(1 / h1_size)
    h1_h2_weights = np.random.randn(h2_size, h1_size) * np.sqrt(1 / h2_size)
    h2_output_weights = np.random.randn(output_size, h2_size) * np.sqrt(1 / output_size)
    return input_h1_weights, h1_h2_weights, h2_output_weights


# Initialise NN biases to random standard normal values
def initialise_biases(h1_size: int, h2_size: int, output_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h1_biases = np.random.randn(h1_size, 1)
    h2_biases = np.random.randn(h2_size, 1)
    output_biases = np.random.randn(output_size, 1)
    return h1_biases, h2_biases, output_biases


# Convert the ubyte pandas df to a numpy array of the data with targets
def ingest_mnist(df: pd.DataFrame, rows: int) -> Tuple[np.ndarray, np.ndarray]:
    input_matrix = np.array(df.iloc[rows, 1:].values.tolist(), ndmin=2).T / 255 # scales by 255 to normalise to 0-1
    targets = np.array(df.iloc[rows, 0].values.tolist(), ndmin=2)
    return input_matrix, targets


# Runs a forward pass using input data, weights, and biases
def forward_propagate(
                      input_matrix: np.ndarray,
                      input_h1_weights: np.ndarray,
                      h1_h2_weights: np.ndarray,
                      h2_output_weights: np.ndarray,
                      h1_biases: np.ndarray,
                      h2_biases: np.ndarray,
                      output_biases: np.ndarray,
                     ) -> np.ndarray:
    h1_z = np.dot(input_h1_weights, input_matrix) + np.tile(h1_biases, input_matrix.shape[1])  # h1_size * batch_size
    h1_a = relu(h1_z)  # h1_size * batch_size
    h2_z = np.dot(h1_h2_weights, h1_a) + np.tile(h2_biases, h1_a.shape[1])  # h2_size * batch_size
    h2_a = relu(h2_z)  # h2_size * batch_size
    output_z = np.dot(h2_output_weights, h2_a) + np.tile(output_biases, h2_a.shape[1])  # output_size * batch size
    output_a = softmax(output_z)  # output_size * batch size
    return output_a, h1_z, h1_a, h2_z, h2_a, output_z


# Runs backpropagation
# Resource 1: https://doug919.github.io/notes-on-backpropagation-with-cross-entropy/
# Resource 2: https://github.com/JohnPaton/numpy-neural-networks/blob/master/02-multi-layer-perceptron.ipynb
def backpropagate(
                  input_matrix: np.ndarray,
                  targets: np.ndarray,
                  learning_rate: float,
                  input_h1_weights: np.ndarray,
                  h1_h2_weights: np.ndarray,
                  h2_output_weights: np.ndarray,
                  h1_biases: np.ndarray,
                  h2_biases: np.ndarray,
                  output_biases: np.ndarray,
                 ) -> Tuple[np.ndarray, ...]:
    target_matrix = np.eye(10)[targets[0]].T

    output_a, h1_z, h1_a, h2_z, h2_a = forward_propagate(
                                                         input_matrix,
                                                         input_h1_weights,
                                                         h1_h2_weights,
                                                         h2_output_weights,
                                                         h1_biases,
                                                         h2_biases,
                                                         output_biases,
                                                        )[:5]

    output_delta_total = np.zeros((1, output_biases.shape[0]))
    h2_delta_total = np.zeros((1, h2_biases.shape[0]))
    h1_delta_total = np.zeros((1, h1_biases.shape[0]))
    output_delta_a = np.zeros((output_biases.shape[0], h2_biases.shape[0]))
    h2_delta_a = np.zeros((h2_biases.shape[0], h1_biases.shape[0]))
    h1_delta_a = np.zeros((h1_biases.shape[0], input_matrix.shape[0]))

    for col in range(input_matrix.shape[1]):
        output_delta = (output_a[:, [col]] - target_matrix[:, [col]]).T  # 1 * output_size
        h2_delta = (np.dot(output_delta, h2_output_weights) * relu_deriv(h2_z[:, [col]]).T)  # 1 * h2_size
        h1_delta = (np.dot(h2_delta, h1_h2_weights) * relu_deriv(h1_z[:, [col]]).T)  # 1 * h1_size
        output_delta_a += np.multiply(output_delta.T, h2_a[:, [col]].T)
        h2_delta_a += np.multiply(h2_delta.T, h1_a[:, [col]].T)
        h1_delta_a += np.multiply(h1_delta.T, input_matrix[:, [col]].T)
        output_delta_total += output_delta
        h2_delta_total += h2_delta
        h1_delta_total += h1_delta

    h2_output_weights -= (learning_rate * 1 / input_matrix.shape[1] * output_delta_a)  # output_size * h2_size
    h1_h2_weights -= (learning_rate * 1 / input_matrix.shape[1] * h2_delta_a)  # h2_size * h1_size
    input_h1_weights -= (learning_rate * 1 / input_matrix.shape[1] * h1_delta_a)  # h1_size * input_size

    output_biases -= learning_rate * output_delta_total.T * 1 / input_matrix.shape[1]
    h2_biases -= learning_rate * h2_delta_total.T * 1 / input_matrix.shape[1]
    h1_biases -= learning_rate * h1_delta_total.T * 1 / input_matrix.shape[1]

    return (input_h1_weights,
            h1_h2_weights,
            h2_output_weights,
            h1_biases,
            h2_biases,
            output_biases
           )


# Trains the network for a defined number of epochs and with batches of a defined size
def train(
          batch_size: int,
          df_train: np.ndarray,
          df_test: np.ndarray,
          epochs: int,
          learning_rate: float,
          input_h1_weights: np.ndarray,
          h1_h2_weights: np.ndarray,
          h2_output_weights: np.ndarray,
          h1_biases: np.ndarray,
          h2_biases: np.ndarray,
          output_biases: np.ndarray,
         ):
    losses = [0] * epochs
    accuracies = [0] * epochs
    for epoch in range(epochs):
        for batch_no in tqdm(range(df_train.shape[0] // batch_size), desc="Epoch " + str(epoch)):
            input_matrix, targets = ingest_mnist(df_train, range(batch_no * batch_size, (batch_no + 1) * batch_size))
            input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases = backpropagate(
                                                                                                                    input_matrix,
                                                                                                                    targets,
                                                                                                                    learning_rate,
                                                                                                                    input_h1_weights,
                                                                                                                    h1_h2_weights,
                                                                                                                    h2_output_weights,
                                                                                                                    h1_biases,
                                                                                                                    h2_biases,
                                                                                                                    output_biases
                                                                                                                   )
        loss, accuracy = test(
                              df_test,
                              input_h1_weights,
                              h1_h2_weights,
                              h2_output_weights,
                              h1_biases,
                              h2_biases,
                              output_biases
                             )
        losses[epoch] = loss
        accuracies[epoch] = accuracy
    return (
        losses,
        accuracies,
        input_h1_weights,
        h1_h2_weights,
        h2_output_weights,
        h1_biases,
        h2_biases,
        output_biases,
    )


def test(
         df: np.ndarray,
         input_h1_weights: np.ndarray,
         h1_h2_weights: np.ndarray,
         h2_output_weights: np.ndarray,
         h1_biases: np.ndarray,
         h2_biases: np.ndarray,
         output_biases: np.ndarray
        ):
    input_matrix, targets = ingest_mnist(df, range(len(df)))
    output_matrix = forward_propagate(
                                      input_matrix,
                                      input_h1_weights,
                                      h1_h2_weights,
                                      h2_output_weights,
                                      h1_biases,
                                      h2_biases,
                                      output_biases,
                                    )[0]
    loss = cat_cross_loss(output_matrix, targets)
    accuracy = np.sum(np.argmax(output_matrix, axis=0) == targets) / len(df)
    return (loss, accuracy)


# Plot the loss and accuracy over epochs on the same graph
def plot_loss_accu(epochs: int, losses: List, accuracies: List):
    ax = plt.gca()
    ax2 = plt.twinx()
    ax.plot(range(epochs), losses, "r")
    ax2.plot(range(epochs), accuracies, "b")
    ax.set_ylabel("Loss", fontsize = 14, color = "r")
    ax2.set_ylabel("Accuracy", fontsize = 14, color = "b")
    ax.set_xlabel("Epoch", fontsize = 14, color = "black")
    plt.title("Loss & Accuracy by Epoch", fontsize = 20, color = "black")
    plt.show()


def main():
    input_size = 784  # DO NOT CHANGE
    h1_size = 1000
    h2_size = 100
    output_size = 10  # DO NOT CHANGE
    learning_rate = 1e-3
    batch_size = 100
    epochs = 5

    print("Loading data...")
    df_train, df_test = read_data()
    print("Data loaded!\n")

    # Initialise weights and biases
    input_h1_weights, h1_h2_weights, h2_output_weights = initialise_weights(
                                                                            input_size,
                                                                            h1_size,
                                                                            h2_size,
                                                                            output_size
                                                                           )
    h1_biases, h2_biases, output_biases = initialise_biases(
                                                            h1_size,
                                                            h2_size,
                                                            output_size
                                                           )

    # Train & Test
    print("Training...")
    losses, accuracies, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases = train(
                                                                                                                        batch_size,
                                                                                                                        df_train,
                                                                                                                        df_test,
                                                                                                                        epochs,
                                                                                                                        learning_rate,
                                                                                                                        input_h1_weights,
                                                                                                                        h1_h2_weights,
                                                                                                                        h2_output_weights,
                                                                                                                        h1_biases,
                                                                                                                        h2_biases,
                                                                                                                        output_biases,
                                                                                                                       )

    # Plot test loss and accuracy over epochs
    plot_loss_accu(epochs, losses, accuracies)
    print("\nFinal epoch loss: " + str(losses[-1]))
    print("Final epoch accuracy: " + str(accuracies[-1]))

    # Export weights and biases to csv for reading in
    print("\nSaving weights/biases...")
    np.savetxt("out/input_h1_weights.csv", input_h1_weights, delimiter=",")
    np.savetxt("out/h1_h2_weights.csv", h1_h2_weights, delimiter=",")
    np.savetxt("out/h2_output_weights.csv", h2_output_weights, delimiter=",")
    np.savetxt("out/h1_biases.csv", h1_biases, delimiter=",")
    np.savetxt("out/h2_biases.csv", h2_biases, delimiter=",")
    np.savetxt("out/output_biases.csv", output_biases, delimiter=",")
    print("Saved!\n")


if __name__ == "__main__":
    main()
