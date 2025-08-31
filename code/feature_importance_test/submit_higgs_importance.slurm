#!/bin/bash
#SBATCH --job-name=higgs_importance_test
#SBATCH --output=higgs_importance_test.out
#SBATCH --error=higgs_importance_test.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=16
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

# Add OpenSSL to the library path to prevent linker errors
export LD_LIBRARY_PATH=${OPENSSL_LIB_DIR}:${LD_LIBRARY_PATH}

CPP_SOURCE_FILE="decision_forest_higgs_importance.cpp"
EXECUTABLE_NAME="higgs_importance_test"

echo "--- COMPILING C++ HIGGS FEATURE IMPORTANCE TEST ---"

g++ -std=c++17 -O3 -fopenmp \
    -I${AOCLDA_ROOT}/include/LP64 \
    -I${LIKWID_ROOT}/include \
    ${CPP_SOURCE_FILE} \
    -L${AOCLDA_ROOT}/lib/LP64  -laocl-da \
    -L${AOCL_ROOT}/lib_LP64   -lalcp -lopenssl-compat \
    -L${OPENSSL_LIB_DIR}       -lssl -lcrypto \
    -L${LIKWID_ROOT}/lib       -llikwid \
    -Wl,-rpath,${AOCLDA_ROOT}/lib/LP64:${AOCL_ROOT}/lib_LP64:${OPENSSL_LIB_DIR}:${LIKWID_ROOT}/lib \
    -lgfortran \
    -o ${EXECUTABLE_NAME}

if [ $? -ne 0 ]; then
    echo "COMPILATION FAILED!"
    exit 1
fi

echo "--- RUNNING EXECUTABLE ---"
./${EXECUTABLE_NAME}
echo "--- JOB COMPLETE ---"
