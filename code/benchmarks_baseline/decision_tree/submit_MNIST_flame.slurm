#!/bin/bash
#SBATCH --job-name=mnist_flame
#SBATCH --output=mnist_flame.out
#SBATCH --error=mnist_flame.err
#SBATCH --partition=test
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module purge
module load gcc/11.2
module load aocl
module load likwid

export OMP_NUM_THREADS=1
export AOCLDA_BIND_THREADS=0

export OPENSSL3_ROOT=/home/wsgp73/openssl3
export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default

export FLAME_HOME=$HOME/FlameGraph

g++ -std=c++17 -g -O3 -fopenmp \
    -I${AOCLDA_ROOT}/include/ILP64 \
    -I${LIKWID_ROOT}/include \
    decision_forest_mnist.cpp \
    ${AOCLDA_ROOT}/lib/ILP64/libaocl-da.so \
    ${AOCL_ROOT}/lib_LP64/libalcp.so \
    ${AOCL_ROOT}/lib_LP64/libopenssl-compat.so \
    ${OPENSSL3_ROOT}/lib64/libcrypto.so.3 \
    ${OPENSSL3_ROOT}/lib64/libssl.so.3 \
    -L${LIKWID_ROOT}/lib -llikwid \
    -lpthread -lm \
    -Wl,-rpath=${AOCLDA_ROOT}/lib/ILP64:${AOCL_ROOT}/lib_LP64:${OPENSSL3_ROOT}/lib64:${LIKWID_ROOT}/lib \
    -o mnist_forest_dbg

perf record -F 99 -g -o perf.data -- ./mnist_forest_dbg

perf script -i perf.data > perf.unfolded
${FLAME_HOME}/stackcollapse-perf.pl perf.unfolded > perf.folded

${FLAME_HOME}/flamegraph.pl perf.folded > flame.svg