#!/bin/bash
#SBATCH --job-name=mnist_cpp_test
#SBATCH --output=mnist_cpp_test.out
#SBATCH --error=mnist_cpp_test.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl
module load likwid

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export OPENSSL3_ROOT=/home/wsgp73/openssl3
export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default

OPENSSL_LIB_DIR=${OPENSSL3_ROOT}/lib64

echo "Using OpenSSL libs in: ${OPENSSL_LIB_DIR}"
echo "--- COMPILING C++ TEST ---"

g++ -std=c++17 -O3 -fopenmp \
    -I${AOCLDA_ROOT}/include/LP64 \
    -I${LIKWID_ROOT}/include \
    decision_forest_mnist.cpp \
    \
    -L${AOCLDA_ROOT}/lib/LP64  -laocl-da \
    -L${AOCL_ROOT}/lib_LP64    -lalcp -lopenssl-compat \
    -L${OPENSSL_LIB_DIR}       -lssl -lcrypto \
    -L${LIKWID_ROOT}/lib       -llikwid \
    \
    -Wl,-rpath-link,${AOCLDA_ROOT}/lib/LP64:${AOCL_ROOT}/lib_LP64:${OPENSSL_LIB_DIR}:${LIKWID_ROOT}/lib \
    -Wl,-rpath,${AOCLDA_ROOT}/lib/LP64:${AOCL_ROOT}/lib_LP64:${OPENSSL_LIB_DIR}:${LIKWID_ROOT}/lib \
    \
    -o mnist_forest_test

if [ $? -ne 0 ]; then
    echo "COMPILATION FAILED!"
    exit 1
fi

echo "--- RUNNING C++ EXECUTABLE ---"
./mnist_forest_test
echo "--- JOB COMPLETE ---"
