#!/usr/bin/env python3

import numpy as np
import time
from sklearn.model_selection import train_test_split
from aoclda.decision_forest import decision_forest

# Load the HIGGS CSV
csv_path = "/nobackup/wsgp73/HIGGS.csv"
data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

# Split into X (features) and y (labels)
X = data[:, :-1]
y = data[:, -1].astype(np.int32)

# Reduce working set for faster training

N_EXAMPLES = 200_000
rng = np.random.default_rng(42)
indices = rng.choice(X.shape[0], size=N_EXAMPLES, replace=False)
X = X[indices]
y = y[indices]

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

n_samples, n_features = X_train.shape

# AOCL-DA Decision Forest
clf = decision_forest(
    n_trees=200,
    max_depth=10,
    min_samples_split=2,
    criterion="entropy",
    seed=42
)

# Fit and time
t0 = time.time()
clf.fit(X_train, y_train)
t1 = time.time()
training_time = t1 - t0

# Evaluate
y_train_pred = clf.predict(X_train)
train_acc = (y_train_pred == y_train).mean()

y_test_pred = clf.predict(X_test)
test_acc = (y_test_pred == y_test).mean()

# Report
print(f"Training time (s): {training_time:.6f}")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
