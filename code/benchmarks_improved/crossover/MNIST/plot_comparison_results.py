# plot_comparison_results.py

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_performance_heatmap():
    
    depth_sweep = [8, 12, 16, 20, 24, 28]
    features_sweep = [16, 24, 32, 48, 64, 96]
    coarse_threads, hybrid_threads = 4, 16
    threads_sweep = [coarse_threads, hybrid_threads]
    results = []
    output_dir = "comparison_results"

    for depth in depth_sweep:
        for features in features_sweep:
            for threads in threads_sweep:
                filename = f"comp_d{depth}_f{features}_t{threads}.out"
                filepath = os.path.join(output_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        match = re.search(r"Training time: (\d+\.\d+) seconds", content)
                        if match:
                            results.append({
                                'depth': depth,
                                'features': features,
                                'threads': threads,
                                'time': float(match.group(1))
                            })
                except FileNotFoundError:
                    pass

    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    pivot_df = df.pivot_table(index=['depth', 'features'], columns='threads', values='time')
    pivot_df.rename(columns={coarse_threads: 'Coarse (4 Threads)', hybrid_threads: 'Hybrid (16 Threads)'}, inplace=True)
    pivot_df['speedup_of_hybrid'] = pivot_df['Coarse (4 Threads)'] / pivot_df['Hybrid (16 Threads)']

    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(12, 9))

    speedup_pivot = pivot_df['speedup_of_hybrid'].unstack(level='features')
    

    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    
    sns.heatmap(
        speedup_pivot,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'label': 'Speedup Ratio (Coarse Time / Hybrid Time)'},
        center=1.0
    )

    ax.set_title(
        'Performance of Hybrid vs. Coarse-Grained Parallelism on MNIST',
        fontsize=18, pad=20, weight='bold'
    )
    ax.set_xlabel('Number of Features Considered per Split', fontsize=14, labelpad=15)
    ax.set_ylabel('Maximum Tree Depth', fontsize=14, labelpad=15)
    
    plt.yticks(rotation=0)

    plt.savefig("performance_heatmap.png", dpi=300, bbox_inches='tight')
    print("\nProfessional performance heatmap saved to performance_heatmap.png")

if __name__ == "__main__":
    generate_performance_heatmap()