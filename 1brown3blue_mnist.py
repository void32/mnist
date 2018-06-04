#!/usr/bin/env python
from load_mnist_data import load_mnist

if __name__ == "__main__":
    train_data, train_labels, train_data, train_labels = load_mnist()
    print("done")