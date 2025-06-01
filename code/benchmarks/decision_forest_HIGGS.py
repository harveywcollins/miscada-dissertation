#!/usr/bin/env python3

import numpy as np
import time

from aoclda.decision_forest import decision_forest

# Load the HIGGS CSV
csv_path = "/nobackup/wsgp73/HIGGS.csv"
data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

# Split into X (features) and y (labels)
X = data[:, :-1]
y = data[:, -1].astype(np.int32)

# Train the AOCL-DA Decision Forest
clf = decision_forest(
    n_estimators=100,
    maximum_depth=15,
    minimum_samples_split=5,
    minimum_samples_leaf=2,
    max_features=0.7,
    seed=42,
    scoring_function="entropy"
)

t0 = time.time()
clf.fit(X, y)
t1 = time.time()
training_time = t1 - t0

# Predict on the same data and compute accuracy
y_pred = clf.predict(X)
accuracy = clf.score(X, y)

# Print training time and accuracy
print(f"Training time (s): {training_time:.6f}")
print(f"Accuracy:         {accuracy:.6f}")

