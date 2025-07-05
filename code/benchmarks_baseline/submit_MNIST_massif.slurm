#!/bin/bash
#SBATCH --job-name=mnist_massif
#SBATCH --output=mnist_massif.out
#SBATCH --error=mnist_massif.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl
module load valgrind

export OPENSSL3_ROOT=/home/wsgp73/openssl3
export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install

g++ -std=c++17 -g -O3 \
    -I${AOCLDA_ROOT}/include/LP64 \
    decision_forest_mnist.cpp \
    ${AOCLDA_ROOT}/lib/LP64/libaocl-da.so \
    ${AOCL_ROOT}/lib_LP64/libalcp.so \
    ${AOCL_ROOT}/lib_LP64/libopenssl-compat.so \
    ${OPENSSL3_ROOT}/lib64/libcrypto.so.3 \
    ${OPENSSL3_ROOT}/lib64/libssl.so.3 \
    -lpthread -lm \
    -Wl,-rpath=${AOCLDA_ROOT}/lib/LP64:${AOCL_ROOT}/lib_LP64:${OPENSSL3_ROOT}/lib64 \
    -o mnist_forest_dbg

valgrind \
  --tool=massif \
  --massif-out-file=massif.out \
  --time-unit=B \
  ./mnist_forest_dbg

ms_print massif.out > massif_report.txt

