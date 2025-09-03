# decision_forest_HIGGS_weak.py

#!/usr/bin/env python3

import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from aoclda.decision_forest import decision_forest

# Load the HIGGS CSV
csv_path = "/nobackup/wsgp73/HIGGS.csv"
data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

# Split into X (features) and y (labels)
X = data[:, :-1]
y = data[:, -1].astype(np.int32)

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Threads & replicate for weak scaling
NT = int(os.environ.get("OMP_NUM_THREADS", "1"))
MAX_THREADS = 128
full_n = X_train.shape[0]
n_train_ws  = (full_n * NT) // MAX_THREADS

X_train_ws = X_train[:n_train_ws]
y_train_ws = y_train[:n_train_ws]

n_samples  = X_train_ws.shape[0]
n_features = X_train_ws.shape[1]

# AOCL-DA Decision Forest
clf = decision_forest(
    n_trees=200,
    max_depth=10,
    min_samples_split=2,
    criterion="cross-entropy",
    seed=42,
    features_selection="custom",
    max_features=int(np.sqrt(n_features))
)

# Fit and time
t0 = time.time()
clf.fit(X_train_ws, y_train_ws)
t1 = time.time()
training_time = t1 - t0

# Evaluate
y_train_pred = clf.predict(X_train)
train_acc = (y_train_pred == y_train).mean()

y_test_pred = clf.predict(X_test)
test_acc = (y_test_pred == y_test).mean()

# Report
print(f"#threads:           {NT}")
print(f"#train samples:    {n_samples}")
print(f"training time [s]: {training_time:.6f}")
print(f"train accuracy:    {train_acc:.4f}")
print(f" test accuracy:    {test_acc:.4f}")
