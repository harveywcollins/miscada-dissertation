# analyze_tuning_results.py

import os
import re
import pandas as pd

def analyze_results():
    """Parses comprehensive comparison results and recommends optimal thresholds."""
    
    depth_sweep = [8, 12, 16, 20, 24, 28]
    features_sweep = [16, 24, 32, 48, 64, 96]
    
    coarse_threads = 4
    hybrid_threads = 16
    threads_sweep = [coarse_threads, hybrid_threads]

    results = []
    output_dir = "comparison_results_full_sweep"

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
                    print(f"Warning: Output file not found: {filepath}")

    if not results:
        print("No results found. Did the benchmark jobs complete successfully?")
        return

    df = pd.DataFrame(results)
    
    pivot_df = df.pivot_table(index=['depth', 'features'], columns='threads', values='time')
    pivot_df.rename(columns={coarse_threads: 'time_coarse', hybrid_threads: 'time_hybrid'}, inplace=True)
    pivot_df['speedup_of_hybrid'] = pivot_df['time_coarse'] / pivot_df['time_hybrid']
    
    print("\n--- Performance Analysis: Coarse-Grained vs. Hybrid Parallelism ---")
    print(pivot_df.round(2))

    # --- Find Optimal Thresholds ---
    print("\n--- Recommended Thresholds ---")
    
    wide_threshold_df = pivot_df.loc[16]
    try:
        recommended_wide = wide_threshold_df[wide_threshold_df['speedup_of_hybrid'] > 1.05].index[0]
        print(f"Recommended WIDE_PROBLEM_THRESHOLD: {recommended_wide}")
        print("(This is the point where parallelism becomes clearly beneficial for 'wide' problems)")
    except IndexError:
        print("Could not determine a clear WIDE_PROBLEM_THRESHOLD. Hybrid was never significantly better.")

    deep_threshold_df = pivot_df.xs(24, level='features')
    try:
        recommended_deep = deep_threshold_df[deep_threshold_df['speedup_of_hybrid'] > 1.05].index[0]
        print(f"Recommended DEEP_PROBLEM_THRESHOLD: {recommended_deep}")
        print("(This is the point where tree depth makes node computation heavy enough for hybrid parallelism to win)")
    except IndexError:
        print("Could not determine a clear DEEP_PROBLEM_THRESHOLD. Hybrid was never significantly better.")

if __name__ == "__main__":
    analyze_results()