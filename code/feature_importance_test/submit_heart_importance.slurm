#!/bin/bash
#SBATCH --job-name=heart_cpp_test
#SBATCH --output=heart_cpp_test.out
#SBATCH --error=heart_cpp_test.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl
module load likwid

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default

export LD_LIBRARY_PATH=${AOCLDA_ROOT}/lib/LP64:${AOCL_ROOT}/lib_LP64:${LIKWID_ROOT}/lib:${LD_LIBRARY_PATH}

echo "--- COMPILING C++ HEART FAILURE TEST ---"

g++ -std=c++17 -O3 -fopenmp \
    -I${AOCLDA_ROOT}/include/LP64 \
    decision_forest_heart.cpp \
    -L${AOCLDA_ROOT}/lib/LP64 -laocl-da \
    -L${LIKWID_ROOT}/lib -llikwid \
    -Wl,-rpath=${AOCLDA_ROOT}/lib/LP64:${LIKWID_ROOT}/lib \
    -o heart_forest_test

if [ $? -ne 0 ]; then
    echo "COMPILATION FAILED!"
    exit 1
fi

echo "--- RUNNING EXECUTABLE ---"
./heart_forest_test
echo "--- JOB COMPLETE ---"
