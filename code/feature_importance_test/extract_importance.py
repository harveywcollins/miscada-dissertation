# extract_importance.py

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mnist_utils import load_mnist_images, load_mnist_labels
from aoclda.decision_forest import decision_forest as random_forest

# Path to MNIST dataset
MNIST_DIR = "/nobackup/wsgp73"
train_images_path = os.path.join(MNIST_DIR, "train-images.idx3-ubyte")
train_labels_path = os.path.join(MNIST_DIR, "train-labels.idx1-ubyte")

# Load only training data
X_train_img = load_mnist_images(train_images_path)
y_train     = load_mnist_labels(train_labels_path)

# Flatten each 28x28 image into a 784-element vector and normalize
X_train = X_train_img.reshape(X_train_img.shape[0], -1).astype(np.float32) / 255.0
y_train = y_train.astype(np.int32)

n_samples, n_features = X_train.shape
print(f"Loaded {n_samples} samples, each with {n_features} features.")

# AOCL-DA Decision Forest Classifier
clf = random_forest(
    n_trees=100,
    max_depth=15,
    min_samples_split=5,
    criterion="entropy",
    seed=42,
    features_selection="custom",
    max_features=int(np.sqrt(n_features))
)

print("Starting model training...")
t0 = time.time()
clf.fit(X_train, y_train)
t1 = time.time()
training_time = t1 - t0
print(f"Training completed in {training_time:.6f} seconds.")

# Feature Importance Extraction 
print("Extracting feature importances...")
feature_importances = clf.get_result("feature_importances")

if feature_importances is not None:
    print(f"Successfully extracted {len(feature_importances)} importance scores.")

    importance_map = feature_importances.reshape(28, 28)

    plt.figure(figsize=(10, 8))
    sns.heatmap(importance_map, cmap="viridis", cbar_kws={'label': 'Importance Score'})
    plt.title("Feature Importance Heatmap for MNIST Digits", fontsize=16)
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    
    output_filename = "feature_importance_heatmap.png"
    plt.savefig(output_filename)
    print(f"Heatmap saved to {output_filename}")

else:
    print("Failed to extract feature importances.")


