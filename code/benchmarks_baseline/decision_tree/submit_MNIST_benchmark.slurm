#!/bin/bash
#SBATCH --job-name=mnist_py_benchmark
#SBATCH --output=mnist_py_benchmark.out
#SBATCH --error=mnist_py_benchmark.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH -c 8

#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl

export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LD_LIBRARY_PATH=$AOCLDA_ROOT/lib/ILP64:$AOCLDA_ROOT/lib/LP64:\
$AOCL_ROOT/lib_LP64:$LD_LIBRARY_PATH
export PYTHONPATH=$AOCLDA_ROOT/python_package:$PYTHONPATH

source /home/wsgp73/miscada-dissertation/code/myenv312/bin/activate
python3 decision_forest_MNIST.py