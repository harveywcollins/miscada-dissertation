#!/usr/bin/env python3

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics       import accuracy_score, roc_curve
from aoclda.decision_forest import decision_forest

# Load the HIGGS data
csv_path = "/nobackup/wsgp73/HIGGS.csv"
data     = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
X, y     = data[:, :-1], data[:, -1].astype(np.int32)

# Train/validation split (80/20)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
n_samples, n_features = X_tr.shape

# Hyperparameter grid
param_grid = {
    "n_trees":           [100, 150, 220],
    "max_depth":         [10, 15, 20],
    "min_samples_split": [2, 4],
    "max_features": [
        int(np.sqrt(n_features)),
        int(0.5 * n_features)
    ]
}

best_score = 0.0
best_cfg   = None

# Grid search
for n_t in param_grid["n_trees"]:
    for md in param_grid["max_depth"]:
        for ms in param_grid["min_samples_split"]:
            for mf in param_grid["max_features"]:
                clf = decision_forest(
                    n_trees=n_t,
                    max_depth=md,
                    min_samples_split=ms,
                    criterion="cross-entropy",
                    seed=42,
                    features_selection="custom",
                    max_features=mf
                )

                # Fit and time
                t0 = time.time()
                clf.fit(X_tr, y_tr)
                train_time = time.time() - t0

                y_pred = clf.predict(X_val)
                proba  = clf.predict_proba(X_val)

                if proba.ndim == 2 and proba.shape[1] >= 2:
                    y_score = proba[:, 1]
                else:
                    y_score = proba.ravel()

                fpr, tpr, _ = roc_curve(y_val, y_score, pos_label=1)
                auc = np.trapz(tpr, fpr)

                acc = accuracy_score(y_val, y_pred)

                if auc > best_score:
                    best_score = auc
                    best_cfg   = (n_t, md, ms, mf, acc, auc, train_time)

# Report best configuration
print("Best config:")
print(" trees, depth, split, features =", best_cfg[:4])
print(f" val acc = {best_cfg[4]:.4f}, AUC = {best_cfg[5]:.4f}")
print(f" train time = {best_cfg[6]:.2f}s")
