#mnist_utils.py

import numpy as np
import struct

# return a Numpy array of shape
def load_mnist_images(idx_filepath):
    with open(idx_filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        return data

# return a Numpy array of shape
def load_mnist_labels(idx_filepath):
    with open(idx_filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
