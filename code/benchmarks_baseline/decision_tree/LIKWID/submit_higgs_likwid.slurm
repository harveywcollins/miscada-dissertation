#!/bin/bash

#SBATCH --job-name=higgs_likwid_profile
#SBATCH --partition=test
#SBATCH --output=higgs_likwid_profile_cache.out
#SBATCH --error=higgs_likwid_profile_cache.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# --- Environment Setup ---
echo "--- SETTING UP ENVIRONMENT ---"
module purge
module load gcc/11.2
module load aocl
module load likwid

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export AOCLDA_BIND_THREADS=1

export OPENSSL3_ROOT=/home/wsgp73/openssl3
export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default
OPENSSL_LIB_DIR=${OPENSSL3_ROOT}/lib64

CPP_SOURCE_FILE="decision_forest_higgs_likwid.cpp"
EXECUTABLE_NAME="higgs_forest_likwid_test"

# --- Compilation ---
echo "--- COMPILING C++ HIGGS TEST for ${OMP_NUM_THREADS} threads ---"
g++ -std=c++17 -O3 -fopenmp -DLIKWID_PERFMON \
    -I${AOCLDA_ROOT}/include/LP64 \
    -I${LIKWID_ROOT}/include \
    ${CPP_SOURCE_FILE} \
    -L${AOCLDA_ROOT}/lib/LP64   -laocl-da \
    -L${AOCL_ROOT}/lib_LP64   -lalcp -lopenssl-compat \
    -L${OPENSSL_LIB_DIR}       -lssl -lcrypto \
    -L${LIKWID_ROOT}/lib       -llikwid \
    -Wl,-rpath-link,${AOCLDA_ROOT}/lib/LP64:${AOCL_ROOT}/lib_LP64:${OPENSSL_LIB_DIR}:${LIKWID_ROOT}/lib \
    -Wl,-rpath,${AOCLDA_ROOT}/lib/LP64:${AOCL_ROOT}/lib_LP64:${OPENSSL_LIB_DIR}:${LIKWID_ROOT}/lib \
    -o ${EXECUTABLE_NAME}

if [ $? -ne 0 ]; then
    echo "COMPILATION FAILED!"
    exit 1
fi
echo "Compilation successful."

echo "--- RUNNING C++ EXECUTABLE WITH LIKWID-PERFCTR ---"
# likwid-perfctr -f -C ${SLURM_CPUS_ON_NODE} -g FLOPS_SP -m ./${EXECUTABLE_NAME}
# likwid-perfctr -f -C ${SLURM_CPUS_ON_NODE} -g MEM -m ./${EXECUTABLE_NAME}
likwid-perfctr -C ${SLURM_CPUS_ON_NODE} -g CACHE -m ./${EXECUTABLE_NAME}
# likwid-perfctr -f -C ${SLURM_CPUS_ON_NODE} -g BRANCH -m ./${EXECUTABLE_NAME}
# likwid-perfctr -f -C ${SLURM_CPUS_ON_NODE} -g TLB -m ./${EXECUTABLE_NAME}
# likwid-perfctr -f -C ${SLURM_CPUS_ON_NODE} -g CLOCK -m ./${EXECUTABLE_NAME}

echo "--- JOB COMPLETE ---"

# --- Cleanup ---
rm ./${EXECUTABLE_NAME}
