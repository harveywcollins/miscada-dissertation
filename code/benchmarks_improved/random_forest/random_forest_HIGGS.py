#!/usr/bin/env python3

import numpy as np
import struct
import os
import time
from mnist_utils import load_mnist_images, load_mnist_labels
from sklearn.model_selection import train_test_split
from aoclda.decision_forest import decision_forest as random_forest

def calculate_accuracy(y_true, y_pred):
    return np.sum(y_pred == y_true) / len(y_true)

if __name__ == "__main__":
    csv_path = "/nobackup/wsgp73/HIGGS_500k.csv"

    print(f"Loading HIGGS dataset from: {csv_path}...")
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
    
    X = data[:, 1:]
    y = data[:, 0].astype(np.int32)
    
    n_samples, n_features = X.shape
    print(f"Dataset loaded: {n_samples} samples, {n_features} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    

    clf = random_forest(
        n_trees=50,
        max_depth=20,
        min_samples_split=10,
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
    
    print("Predicting on training set...")
    y_train_pred = clf.predict(X_train)
    train_accuracy = calculate_accuracy(y_train, y_train_pred)
    
    print("Predicting on test set...")
    y_test_pred = clf.predict(X_test)
    test_accuracy = calculate_accuracy(y_test, y_test_pred)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:     {test_accuracy:.4f}")