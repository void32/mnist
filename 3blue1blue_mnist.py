#!/usr/bin/env python
"""
Here I  will try to implement the simple and probbly not that good network from  the 1brown3blue
[But what *is* a Neural Network? | Chapter 1, deep learning | https://www.youtube.com/watch?v=aircAruvnKk] series

The pupose is not to make a good neural network, or to explore the types of layers, but simply to get handson with
Tensorflow basics and the Math behide CNNs.
"""

import tensorflow as tf
from load_mnist_data import IMAGE_SIZE, NUM_LABELS, load_mnist

# This defines the size of the batch - (We'll bundle groups of examples during training for efficiency).
BATCH_SIZE = 60
# We have only one channel in our grayscale images.
NUM_CHANNELS = 1
# The random seed that defines initialization.
SEED = 42


if __name__ == "__main__":
    """ Load Data set """
    train_data, train_labels, train_data, train_labels = load_mnist()

    """ Define variables in the model, these will hold the trainable weights """
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each training step.
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))


    """ Define structure of the basic model graph """




    # Train ...

    # Use the test set to predict and calculate error rate

    # visualize the results for the test set and the layers trained

