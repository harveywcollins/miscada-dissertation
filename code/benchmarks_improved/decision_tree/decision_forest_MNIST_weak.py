# decision_forest_MNIST_weak.py

import os
import time
import numpy as np
from mnist_utils import load_mnist_images, load_mnist_labels
from aoclda.decision_forest import decision_forest

# Path to MNIST dataset
MNIST_DIR         = "/nobackup/wsgp73"
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
X_train = (X_train_img.reshape(len(X_train_img), -1).astype(np.float32) / 255.0)
X_test  = (X_test_img.reshape(len(X_test_img),  -1).astype(np.float32) / 255.0)
y_train = y_train.astype(np.int32)
y_test  = y_test.astype(np.int32)

# Threads & replicate for weak scaling
NT = int(os.environ.get("OMP_NUM_THREADS", "1"))

MAX_THREADS = 128
full_n = X_train.shape[0]
n_train_ws = full_n * NT // MAX_THREADS

X_train_ws = X_train[:n_train_ws]
y_train_ws = y_train[:n_train_ws]

n_samples  = X_train_ws.shape[0]
n_features = X_train_ws.shape[1]

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
clf.fit(X_train_ws, y_train_ws)
t1 = time.time()
training_time = t1 - t0

# Evaluate accuracy on train and test sets
y_train_pred = clf.predict(X_train)
y_test_pred  = clf.predict(X_test)
train_acc    = (y_train_pred == y_train).mean()
test_acc     = (y_test_pred  == y_test).mean()

# Report
print(f"#threads: {NT}")
print(f"#train samples: {n_samples}")
print(f"training time [s]: {training_time:.6f}")
print(f"train accuracy: {train_acc:.4f}")
print(f" test accuracy: {test_acc:.4f}")

