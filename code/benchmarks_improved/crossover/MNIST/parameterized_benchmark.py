# parameterized_benchmark.py

import time
import numpy as np
import os
import argparse

from mnist_utils import load_mnist_images, load_mnist_labels
from aoclda.decision_forest import decision_forest

def run_benchmark(n_trees, max_depth, n_features):
    """
    Trains a random forest to compare coarse-grained vs. hybrid parallelism.
    """
    threads = os.getenv('OMP_NUM_THREADS', 'N/A')
    print(f"--- Benchmark: trees={n_trees}, depth={max_depth}, features={n_features}, threads={threads} ---")
    
    MNIST_DIR = "/nobackup/wsgp73"
    train_images_path = os.path.join(MNIST_DIR, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(MNIST_DIR, "train-labels.idx1-ubyte")
    X_train_img = load_mnist_images(train_images_path)
    y_train     = load_mnist_labels(train_labels_path)
    X_train = X_train_img.reshape(X_train_img.shape[0], -1).astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)

    clf = decision_forest(
        n_trees=n_trees,
        max_depth=max_depth,
        features_selection="custom",
        max_features=n_features,
        seed=42
    )
    
    start_time = time.perf_counter()
    clf.fit(X_train, y_train)
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"Training time: {duration:.6f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a random forest benchmark.")
    parser.add_argument('--trees', type=int, required=True, help='Number of trees in the forest.')
    parser.add_argument('--depth', type=int, required=True, help='Maximum depth of the trees.')
    parser.add_argument('--features', type=int, required=True, help='Number of features to consider for a split.')
    args = parser.parse_args()

    run_benchmark(args.trees, args.depth, args.features)