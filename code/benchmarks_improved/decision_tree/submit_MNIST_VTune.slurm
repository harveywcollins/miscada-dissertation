#!/bin/bash
#SBATCH --job-name=mnist_vtune
#SBATCH --output=mnist_vtune.out
#SBATCH --error=mnist_vtune.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl
module load likwid
module load vtune/2024.2

export OPENSSL3_ROOT=/home/wsgp73/openssl3
export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default

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

rm -rf vtune/vtune_mem_mnist vtune/vtune_hotspots_mnist

vtune -collect memory-consumption \
      -result-dir vtune/vtune_mem_mnist \
      -- ./mnist_forest_dbg

vtune -report summary \
      -result-dir vtune/vtune_mem_mnist \
      -filter module=libaocl-da.so \
      -format text \
    > vtune/vtune_mem_mnist_lib_summary.txt

vtune -report top-down \
      -result-dir vtune/vtune_mem_mnist \
      -filter module=libaocl-da.so \
      -format text \
    > vtune/vtune_mem_mnist_lib_topdown.txt

vtune -report timeline \
      -result-dir vtune/vtune_mem_mnist \
      -filter module=libaocl-da.so \
      -format text \
    > vtune/vtune_mem_mnist_lib_timeline.txt

vtune -collect hotspots \
      -result-dir vtune/vtune_hotspots_mnist \
      -- ./mnist_forest_dbg

vtune -report summary \
      -result-dir vtune/vtune_hotspots_mnist \
      -filter module=libaocl-da.so \
      -format text \
    > vtune/vtune_hotspots_lib_summary.txt

vtune -report top-down \
      -result-dir vtune/vtune_hotspots_mnist \
      -filter module=libaocl-da.so \
      -format text \
    > vtune/vtune_hotspots_lib_topdown.txt