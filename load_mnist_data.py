from __future__ import print_function

import os
from six.moves.urllib.request import urlretrieve
import gzip
import numpy

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = "/tmp/mnist-data"

"""
# Working with the images from the MINST data set

The data is gzipped, requiring us to decompress it. And, each of the images are
grayscale-encoded with values from [0, 255]; we'll normalize these to [-0.5, 0.5].

Unpack the data using the documented format:

    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white),
255 means foreground (black).

# Reading the labels from the MINST data set

Unpack the test label data. The format here is similar: a magic number followed by a
count followed by the labels as `uint8` values. In more detail:

    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label

As with the image data, let's read the first test set value to sanity check our input path.
We'll expect a 7.
"""
NUM_LABELS = 10  # The number of label for minst is the ten digitst 0,1,..,9,10
IMAGE_SIZE = 28  # number of rows and columns
PIXEL_DEPTH = 255  # 8 bit

NUM_TRAIN_IMAGES = 60000
NUM_TEST_IMAGES = 10000

def maybe_download(filename):
    """A helper to download the data files if not present."""
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
    return filepath

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    For MNIST data, the number of channels is always 1.

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)

        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and count; we know these values.
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)


def load_mnist_data(train_data_filename, test_data_filename):
    train_data = extract_data(train_data_filename, NUM_TRAIN_IMAGES)
    test_data = extract_data(test_data_filename, NUM_TEST_IMAGES)
    return train_data, test_data


def load_mnist_labels(train_labels_filename, test_labels_filename):
    train_labels = extract_labels(train_labels_filename, NUM_TRAIN_IMAGES)
    test_labels = extract_labels(test_labels_filename, NUM_TEST_IMAGES)
    return train_labels, test_labels


def load_mnist():
    # Download the data archives
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Load data and images
    train_data, test_data = load_mnist_data(train_data_filename, test_data_filename)
    train_labels, test_labels = load_mnist_labels(train_labels_filename, test_labels_filename)

    return train_data, train_labels, train_data, train_labels
