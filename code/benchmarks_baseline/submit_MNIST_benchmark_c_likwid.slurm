#!/bin/bash
#SBATCH --job-name=mnist_forest_likwid
#SBATCH --output=mnist_cpp_benchmark_likwid_MEM.out
#SBATCH --error=mnist_cpp_benchmark_likwid_MEM.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl
module load likwid

export AOCLDA_BIND_THREADS=1
export OMP_NUM_THREADS=8

export OPENSSL3_ROOT=/home/wsgp73/openssl3
export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install

export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default

g++ -std=c++17 -O3 -fopenmp -DLIKWID_PERFMON\
    -I${AOCLDA_ROOT}/include/LP64 \
    -I${LIKWID_ROOT}/include \
    decision_forest_mnist_likwid.cpp \
    ${AOCLDA_ROOT}/lib/LP64/libaocl-da.so \
    ${AOCL_ROOT}/lib_LP64/libalcp.so \
    ${AOCL_ROOT}/lib_LP64/libopenssl-compat.so \
    ${OPENSSL3_ROOT}/lib64/libcrypto.so.3 \
    ${OPENSSL3_ROOT}/lib64/libssl.so.3 \
    -L${LIKWID_ROOT}/lib -llikwid \
    -lpthread -lm \
    -Wl,-rpath=${AOCLDA_ROOT}/lib/LP64:${AOCL_ROOT}/lib_LP64:${OPENSSL3_ROOT}/lib64:${LIKWID_ROOT}/lib \
    -o mnist_forest_likwid

# likwid-perfctr -f -C 0-7 -g CACHE -m ./mnist_forest_likwid
likwid-perfctr -f -C 0-7 -g MEM -m ./mnist_forest_likwid