#!/bin/bash
#SBATCH --job-name=mnist_feature_importance
#SBATCH --output=MNIST_importance/mnist_importance.out
#SBATCH --error=MNIST_importance/mnist_importance.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH -c 16

#SBATCH --mem=32G

mkdir -p MNIST_importance

module purge
module load gcc/11.2
module load aocl
module load likwid

export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default

export LD_PRELOAD=$LIKWID_ROOT/lib/liblikwid.so

export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LD_LIBRARY_PATH=$AOCLDA_ROOT/lib/ILP64:\
$AOCLDA_ROOT/lib/LP64:\
$AOCL_ROOT/lib_LP64:\
$AOCL_ROOT/lib_ILP64:\
$AOCL_ROOT/lib:\
$LIKWID_ROOT/lib:\
$LD_LIBRARY_PATH

export PYTHONPATH=$AOCLDA_ROOT/python_package:$PYTHONPATH

export OMP_NUM_THREADS=16
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export KMP_AFFINITY=verbose,granularity=fine,scatter

source /home/wsgp73/miscada-dissertation/code/myenv312/bin/activate
numactl --interleave=all python3 extract_importance.py

