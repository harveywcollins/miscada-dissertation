#!/usr/bin/env python3

import numpy as np
import struct
import os
import time
from mnist_utils import load_mnist_images, load_mnist_labels
from aoclda.decision_forest import decision_forest as random_forest


def load_data(data_path, labels_path):
    X = np.loadtxt(data_path, dtype=np.float32)
    y = np.loadtxt(labels_path, dtype=np.int32)
    return X, y

def calculate_accuracy(y_true, y_pred):
    return np.sum(y_pred == y_true) / len(y_true)

if __name__ == "__main__":
    base_path = "/nobackup/wsgp73/ARCENE/"
    train_data_path = os.path.join(base_path, "arcene_train.data")
    train_labels_path = os.path.join(base_path, "arcene_train.labels")
    
    test_data_path = os.path.join(base_path, "arcene_valid.data")
    test_labels_path = "/nobackup/wsgp73/arcene_valid.labels" # Path from your ls output

    print("Loading Arcene dataset...")
    X_train, y_train_raw = load_data(train_data_path, train_labels_path)
    X_test, y_test_raw = load_data(test_data_path, test_labels_path)
    
    y_train = np.copy(y_train_raw)
    y_test = np.copy(y_test_raw)
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    
    n_samples, n_features = X_train.shape
    print(f"Dataset loaded: {n_samples} training samples, {n_features} features.")
    
    clf = random_forest(
        n_trees=64,
        max_depth=30,
        min_samples_split=2,
        criterion="entropy",
        seed=42,
        features_selection="custom",
        max_features=int(np.sqrt(n_features))
    )

    print("Starting model training...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.6f} seconds.")
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_accuracy = calculate_accuracy(y_train, y_train_pred)
    test_accuracy = calculate_accuracy(y_test, y_test_pred)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:     {test_accuracy:.4f}")