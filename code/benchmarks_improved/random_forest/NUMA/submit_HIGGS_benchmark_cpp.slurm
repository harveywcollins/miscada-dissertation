#!/bin/bash

OUTPUT_DIR="HIGGS_strongscaling_cpp_NUMA"
mkdir -p ${OUTPUT_DIR}

CPP_SOURCE_FILE="random_forest_higgs.cpp"

if [ ! -f "${CPP_SOURCE_FILE}" ]; then
    echo "ERROR: C++ source file '${CPP_SOURCE_FILE}' not found."
    echo "Please ensure the C++ runner file is in the same directory as this script."
    exit 1
fi

for NT in 1 2 4 8 16 32 64 128; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=higgs_cpp_${NT}
#SBATCH --output=${OUTPUT_DIR}/higgs_strong_cpp_${NT}.out
#SBATCH --error=${OUTPUT_DIR}/higgs_strong_cpp_${NT}.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NT}
#SBATCH --mem=32G

# Purge and load necessary modules for the compilation and execution environment.
module purge
module load gcc/11.2
module load aocl
module load likwid

# Set environment variables required by the application and its libraries.
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OPENSSL3_ROOT=/home/wsgp73/openssl3
export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default
OPENSSL_LIB_DIR=\${OPENSSL3_ROOT}/lib64

EXECUTABLE_NAME="higgs_forest_test_\${OMP_NUM_THREADS}"

echo "--- COMPILING C++ HIGGS TEST for \${OMP_NUM_THREADS} threads ---"

export LD_LIBRARY_PATH=\${OPENSSL_LIB_DIR}:\${LD_LIBRARY_PATH}

g++ -std=c++17 -O3 -fopenmp \\
    -I\${AOCLDA_ROOT}/include/LP64 \\
    -I\${LIKWID_ROOT}/include \\
    ${CPP_SOURCE_FILE} \\
    -L\${AOCLDA_ROOT}/lib/LP64  -laocl-da \\
    -L\${AOCL_ROOT}/lib_LP64   -lalcp -lopenssl-compat \\
    -L\${OPENSSL_LIB_DIR}       -lssl -lcrypto \\
    -L\${LIKWID_ROOT}/lib       -llikwid \\
    -Wl,-rpath-link,\${AOCLDA_ROOT}/lib/LP64:\${AOCL_ROOT}/lib_LP64:\${OPENSSL_LIB_DIR}:\${LIKWID_ROOT}/lib \\
    -Wl,-rpath,\${AOCLDA_ROOT}/lib/LP64:\${AOCL_ROOT}/lib_LP64:\${OPENSSL_LIB_DIR}:\${LIKWID_ROOT}/lib \\
    -o \${EXECUTABLE_NAME}

# Check if compilation was successful.
if [ \$? -ne 0 ]; then
    echo "COMPILATION FAILED!"
    exit 1
fi

echo "--- RUNNING C++ EXECUTABLE ---"
echo "Applying memory interleave policy across all NUMA nodes."
numactl --interleave=all ./\${EXECUTABLE_NAME}

echo "--- JOB COMPLETE ---"

# Clean up the executable after the run.
rm ./\${EXECUTABLE_NAME}

EOF
done

echo "Submitted all HIGGS C++ benchmark jobs to SLURM."
