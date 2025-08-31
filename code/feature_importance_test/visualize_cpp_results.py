import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

print("Running C++ executable to get feature importances...")

aoclda_root = "/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install"
aocl_root = "/home/wsgp73/aocl/5.0.0/gcc"
likwid_root = "/apps/developers/libraries/likwid/5.2.0/1/default"

gcc_lib_path = "/apps/developers/compilers/gcc/11.2/1/default/lib64"

env = os.environ.copy()


env['LD_LIBRARY_PATH'] = (
    f"{gcc_lib_path}:"
    f"{aoclda_root}/lib/LP64:"
    f"{aocl_root}/lib_LP64:"
    f"{likwid_root}/lib:"
    + env.get('LD_LIBRARY_PATH', '')
)

print(f"Using LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")

try:
    process = subprocess.run(
        ['./mnist_forest_test'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
        env=env
    )
except subprocess.CalledProcessError as e:
    print("\n--- C++ EXECUTABLE FAILED ---")
    print("The C++ program exited with an error.")
    print(f"Return Code: {e.returncode}")
    print("\n----- C++ STDOUT -----")
    print(e.stdout)
    print("\n----- C++ STDERR -----")
    print(e.stderr)
    exit(1)


output_text = process.stdout
print("C++ executable finished successfully.")

importances = []
found_scores_line = False
for line in output_text.splitlines():
    if "Successfully extracted" in line and "importance scores" in line:
        found_scores_line = True
        continue
    
    if found_scores_line and "Feature" in line:
        match = re.search(r'Feature\s+\d+:\s+([0-9.]+)', line)
        if match:
            importances.append(float(match.group(1)))

if not importances:
    print("Could not find any feature importance scores in the C++ output.")
    exit(1)

print(f"Successfully parsed {len(importances)} importance scores.")

importances = np.array(importances)
most_important_pixel_index = np.argmax(importances)
highest_score = np.max(importances)

print(f"\nMost important feature (pixel): #{most_important_pixel_index}")
print(f"Highest importance score: {highest_score:.6f}")

if len(importances) == 784:
    importance_map = importances.reshape(28, 28)

    plt.figure(figsize=(10, 8))
    sns.heatmap(importance_map, cmap="viridis", cbar_kws={'label': 'Importance Score'})
    plt.title("Feature Importance Heatmap from C++ Results", fontsize=16)
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    
    output_filename = "cpp_feature_importance_heatmap.png"
    plt.savefig(output_filename)
    print(f"\nHeatmap visualization saved to '{output_filename}'")
else:
    print(f"Warning: Expected 784 features for a 28x28 reshape, but found {len(importances)}.")

