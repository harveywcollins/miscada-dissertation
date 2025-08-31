#decision_forest_MNIST.py

import numpy as np
import struct
import os
import time
from mnist_utils import load_mnist_images, load_mnist_labels
from aoclda.decision_forest import decision_forest

# Path to MNIST dataset
MNIST_DIR = "/nobackup/wsgp73"
train_images_path = os.path.join(MNIST_DIR, "train-images.idx3-ubyte")
train_labels_path = os.path.join(MNIST_DIR, "train-labels.idx1-ubyte")
test_images_path  = os.path.join(MNIST_DIR, "t10k-images.idx3-ubyte")
test_labels_path  = os.path.join(MNIST_DIR, "t10k-labels.idx1-ubyte")

# Load train and test data
X_train_img = load_mnist_images(train_images_path)
y_train     = load_mnist_labels(train_labels_path)

X_test_img  = load_mnist_images(test_images_path)
y_test      = load_mnist_labels(test_labels_path)

# Flatten each 28x28 image into vector
X_train = X_train_img.reshape(X_train_img.shape[0], -1).astype(np.float32) / 255.0
X_test  = X_test_img.reshape(X_test_img.shape[0],  -1).astype(np.float32) / 255.0

y_train = y_train.astype(np.int32)
y_test  = y_test.astype(np.int32)

n_samples, n_features = X_train.shape

# AOCL-DA Decision Forest
clf = decision_forest(
    n_trees=100,
    max_depth=15,
    min_samples_split=5,
    criterion="entropy",
    seed=42,
    features_selection="custom",
    max_features=int(np.sqrt(n_features))
)

t0 = time.time()
clf.fit(X_train, y_train)
t1 = time.time()
training_time = t1 - t0
print(f"Training completed in {training_time:.6f} seconds.")

# Evaluate accuracy on train and test sets
y_train_pred = clf.predict(X_train)
train_acc = (y_train_pred == y_train).mean()
y_test_pred  = clf.predict(X_test)
test_acc = (y_test_pred == y_test).mean()

print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy:     {test_acc:.4f}")
