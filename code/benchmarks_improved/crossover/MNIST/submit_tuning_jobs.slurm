#!/bin/bash

mkdir -p comparison_results

DEPTH_SWEEP="8 12 16 20 24 28"
FEATURES_SWEEP="16 24 32 48 64 96"

NUM_TREES=4
COARSE_THREADS=$NUM_TREES 
HYBRID_THREADS=16

echo "Submitting comprehensive comparison jobs..."
echo "This will submit $(echo "$DEPTH_SWEEP" | wc -w) * $(echo "$FEATURES_SWEEP" | wc -w) * 2 = 72 jobs."

for depth in $DEPTH_SWEEP; do
  for features in $FEATURES_SWEEP; do
  
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=comp_d${depth}_f${features}_t${COARSE_THREADS}
#SBATCH --output=comparison_results_full_sweep/comp_d${depth}_f${features}_t${COARSE_THREADS}.out
#SBATCH --error=comparison_results_full_sweep/comp_d${depth}_f${features}_t${COARSE_THREADS}.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${COARSE_THREADS}
#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl
module load likwid
module load tbb

export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default

export LD_PRELOAD=\$LIKWID_ROOT/lib/liblikwid.so

export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LD_LIBRARY_PATH=\$AOCLDA_ROOT/lib/ILP64:\
\$AOCLDA_ROOT/lib/LP64:\
\$AOCL_ROOT/lib_LP64:\
\$AOCL_ROOT/lib_ILP64:\
\$AOCL_ROOT/lib:\
\$LIKWID_ROOT/lib:\
\$LD_LIBRARY_PATH

export PYTHONPATH=\$AOCLDA_ROOT/python_package:\$PYTHONPATH

export OMP_NUM_THREADS=${COARSE_THREADS}
export OMP_PROC_BIND=close; export OMP_PLACES=cores
source /home/wsgp73/miscada-dissertation/code/myenv312/bin/activate

python3 parameterized_benchmark.py --trees ${NUM_TREES} --depth ${depth} --features ${features}
EOF

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=comp_d${depth}_f${features}_t${HYBRID_THREADS}
#SBATCH --output=comparison_results_full_sweep/comp_d${depth}_f${features}_t${HYBRID_THREADS}.out
#SBATCH --error=comparison_results_full_sweep/comp_d${depth}_f${features}_t${HYBRID_THREADS}.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${HYBRID_THREADS}
#SBATCH --mem=32G
# --- Environment Setup ---
module purge
module load gcc/11.2
module load aocl
module load likwid
module load tbb

export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default

export LD_PRELOAD=\$LIKWID_ROOT/lib/liblikwid.so

export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LD_LIBRARY_PATH=\$AOCLDA_ROOT/lib/ILP64:\
\$AOCLDA_ROOT/lib/LP64:\
\$AOCL_ROOT/lib_LP64:\
\$AOCL_ROOT/lib_ILP64:\
\$AOCL_ROOT/lib:\
\$LIKWID_ROOT/lib:\
\$LD_LIBRARY_PATH

export PYTHONPATH=\$AOCLDA_ROOT/python_package:\$PYTHONPATH

export OMP_NUM_THREADS=${HYBRID_THREADS}
export OMP_PROC_BIND=close; export OMP_PLACES=cores
source /home/wsgp73/miscada-dissertation/code/myenv312/bin/activate

python3 parameterized_benchmark.py --trees ${NUM_TREES} --depth ${depth} --features ${features}
EOF

  done
done

echo "All comparison jobs submitted."
