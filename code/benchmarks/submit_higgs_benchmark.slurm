#!/bin/bash
#SBATCH --job-name=higgs_py_benchmark
#SBATCH --output=higgs_py_benchmark.out
#SBATCH --error=higgs_py_benchmark.err
#SBATCH --partition=shared
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load gcc/11.2

export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LD_LIBRARY_PATH=$AOCLDA_ROOT/lib/LP64:$LD_LIBRARY_PATH

source /home/wsgp73/miscada-dissertation/code/myenv312/bin/activate

python3 decision_forest_HIGGS.py

