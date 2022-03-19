# --------------------------------------------------------------------------- #
# FILE:     play_with_trained.py                                              #
#                                                                             #
# PURPOSE:  read in network from .csv's and classify images                   #
# --------------------------------------------------------------------------- #

from mnist import MNIST
from matplotlib import pyplot as plt
import numpy as np
import random

fashion_dict = {'0': 'Top', '1': 'Trouser', '2': 'Pullover',
                '3': 'Dress', '4': 'Coat', '5': 'Sandal',
                '6': 'Shirt', '7': 'Sneaker', '8': 'Bag', '9': 'Boot'}

mnist = MNIST('fashion_data')
images, labels = mnist.load_training()

# Import saved params
input_h1_weights = np.genfromtxt("out/input_h1_weights.csv", delimiter=",")
h1_h2_weights = np.genfromtxt("out/h1_h2_weights.csv", delimiter=",")
h2_output_weights = np.genfromtxt("out/h2_output_weights.csv", delimiter=",")
h1_biases = np.array(np.genfromtxt("out/h1_biases.csv", delimiter=","), ndmin=2).T
h2_biases = np.array(np.genfromtxt("out/h2_biases.csv", delimiter=","), ndmin=2).T
output_biases = np.array(np.genfromtxt("out/output_biases.csv", delimiter=","), ndmin=2).T


# Functions to forward_propagate
def relu(z):
    return np.where(z > 0, z, 0.01 * z)

def softmax(vector):
    e = np.exp(vector - np.max(vector))
    return e / e.sum()

def forward_propagate(input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases):
    input_vector = input_with_target[0]
    h1_vector = relu(np.dot(input_h1_weights, input_vector) + h1_biases)
    h2_vector = relu(np.dot(h1_h2_weights, h1_vector) + h2_biases)
    output_vector = softmax(np.dot(h2_output_weights, h2_vector) + output_biases)
    return output_vector


# Loop to keep testing until ctrl-c
while True:

    # prep image and NN data
    index = random.randrange(0, len(images))
    image_to_show = np.array(images[index], dtype='float')
    pixels = image_to_show.reshape((28, 28))
    input_with_target = (np.array(images[index], ndmin=2).T, labels[index])

    # output vector from NN prediction
    output_vector = forward_propagate(input_with_target, input_h1_weights, h1_h2_weights, h2_output_weights, h1_biases, h2_biases, output_biases)

    # compare actual to NN classification; show image
    print("\nActual classification")
    print(str(labels[index])+": "+fashion_dict[str(labels[index])])
    print(output_vector)
    print("\nFFMLP classification:")
    print(str(np.argmax(output_vector)) + ": " + fashion_dict[str(np.argmax(output_vector))])
    
    # plot
    plt.imshow(pixels, cmap='gray_r')
    plt.title("Actual: " + fashion_dict[str(labels[index])] + " ~ Predicted: " + fashion_dict[str(np.argmax(output_vector))] + " ~ Confidence: " + str(round(np.amax(output_vector), 2)))
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()