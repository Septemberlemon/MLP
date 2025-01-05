import struct

import numpy as np


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Magic number mismatch: {magic}"
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)  # (num_images, 28, 28)
        return images


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Magic number mismatch: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


# 使用示例
train_images = load_mnist_images("MNIST/raw/train-images-idx3-ubyte")
train_labels = load_mnist_labels("MNIST/raw/train-labels-idx1-ubyte")

test_images = load_mnist_images("MNIST/raw/t10k-images-idx3-ubyte")
test_labels = load_mnist_labels("MNIST/raw/t10k-labels-idx1-ubyte")

train_data = [(np.reshape(x / 255, (28 * 28, 1)), y) for x, y in zip(train_images, train_labels)]

test_data = [(np.reshape(x / 255, (28 * 28, 1)), y) for x, y in zip(test_images, test_labels)]
