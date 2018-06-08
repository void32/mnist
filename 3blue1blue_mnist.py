#!/usr/bin/env python
"""
Here I  will try to implement the simple and probbly not that good network from  the 1brown3blue
[But what *is* a Neural Network? | Chapter 1, deep learning | https://www.youtube.com/watch?v=aircAruvnKk] series

The pupose is not to make a good neural network, or to explore the types of layers, but simply to get handson with
Tensorflow basics and the Math behide CNNs.
"""

import tensorflow as tf
from load_mnist_data import IMAGE_SIZE, IMAGE_PIXELS, NUM_LABELS, load_mnist

# This defines the size of the batch - (We'll bundle groups of examples during training for efficiency).
BATCH_SIZE = 60
# We have only one channel in our grayscale images.
NUM_CHANNELS = 1
# The random seed that defines initialization.
SEED = 42


if __name__ == "__main__":
    """ Load data set """
    train_data, train_labels, test_data, test_labels = load_mnist()


    """ Build the model

    The architecture used in the 3blue1brown lesson https://youtu.be/aircAruvnKk?t=2m3s

    Variable              Operations               Variable            Operations              Variable              Operations               Variable
                       matmul,addition,relu                        matmul,addition,relu                           matmul,addition,relu
    Input layer   ----------748x16----------> First hidden layer ---------16x16---------> Second hidden layer ------------16x10-----------> Output layer
    (784 neurons)      (Fully connected)        (16 neurons)          (Fully connected)       (16 neurons)           (Fully connected)       (16 neurons)
    Pixel values                                                                                                                             Labels: 0 1 .. 9

    """

    """ Define variables in the model, these will hold the trainable weights """


    # This is where training samples and labels are fed to the graph
    # (we feed data into the graph through these placeholders)
    # These placeholder nodes will be fed a batch of training data at each training step.
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))

    # Regarding the test data, the entire dataset is held in memory as one constant node
    test_data_node = tf.constant(test_data)

    # Parameters/Variables for the hidden layers:
    # The variables below hold all the trainable weights. For each, the
    # parameter defines how the variables will be initialized.

    # First hidden layer
    NEURONS_HIDDEN_LAYER_1 = 16
    fc1_weights = tf.Variable(
      tf.truncated_normal(
        shape=[IMAGE_PIXELS, NEURONS_HIDDEN_LAYER_1],
        stddev=0.1,
        seed=SEED
      )
    )
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[NEURONS_HIDDEN_LAYER_1]))


    # Second hidden layer
    NEURONS_HIDDEN_LAYER_2 = 16
    fc2_weights = tf.Variable(
      tf.truncated_normal(
        shape=[NEURONS_HIDDEN_LAYER_1, NEURONS_HIDDEN_LAYER_2],
        stddev=0.1,
        seed=SEED
      )
    )
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NEURONS_HIDDEN_LAYER_2]))


    # Result layer
    fc2_weights = tf.Variable(
      tf.truncated_normal(
        shape=[NEURONS_HIDDEN_LAYER_2, NUM_LABELS],
        stddev=0.1,
        seed=SEED
      )
    )
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


    """ Define structure of the basic model graph """

    """ Train ... """

    """ Use the test set to predict and calculate error rate """

    """ visualize the results for the test set and the layers trained  """

