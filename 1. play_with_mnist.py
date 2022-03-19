# --------------------------------------------------------------------------- #
# FILE:     play_with_mnist.py                                                #
#                                                                             #
# PURPOSE:  play with the mnist library to print and explore the data         #
# --------------------------------------------------------------------------- #

from mnist import MNIST
from matplotlib import pyplot as plt
import numpy as np
import random

# import data
mnist = MNIST('fashion_data')
images, labels = mnist.load_training()

# choose image to show
index = random.randrange(0, len(images))
image_to_show = np.array(images[index], dtype='float')
pixels = image_to_show.reshape((28, 28))

# dict for image numbers to names
fashion_dict = {'0': 'Top', '1': 'Trouser', '2': 'Pullover',
                '3': 'Dress', '4': 'Coat', '5': 'Sandal',
                '6': 'Shirt', '7': 'Sneaker', '8': 'Bag', '9': 'Boot'}

# show image
print(str(labels[index])+": "+fashion_dict[str(labels[index])])
plt.imshow(pixels, cmap='gray_r')
plt.title(fashion_dict[str(labels[index])])
plt.show()