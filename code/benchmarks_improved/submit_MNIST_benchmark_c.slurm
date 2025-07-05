#!/bin/bash
#SBATCH --job-name=mnist_forest
#SBATCH --output=mnist_cpp_benchmark.out
#SBATCH --error=mnist_cpp_benchmark.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl

export AOCLDA_BIND_THREADS=1
export OMP_NUM_THREADS=8

export OPENSSL3_ROOT=/home/wsgp73/openssl3
export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install

g++ -std=c++17 -O3 \
    -I${AOCLDA_ROOT}/include/LP64 \
    decision_forest_mnist.cpp \
    ${AOCLDA_ROOT}/lib/LP64/libaocl-da.so \
    ${AOCL_ROOT}/lib_LP64/libalcp.so \
    ${AOCL_ROOT}/lib_LP64/libopenssl-compat.so \
    ${OPENSSL3_ROOT}/lib64/libcrypto.so.3 \
    ${OPENSSL3_ROOT}/lib64/libssl.so.3 \
    -lpthread -lm \
    -Wl,-rpath=${AOCLDA_ROOT}/lib/LP64:${AOCL_ROOT}/lib_LP64:${OPENSSL3_ROOT}/lib64 \
    -o mnist_forest

./mnist_forest
