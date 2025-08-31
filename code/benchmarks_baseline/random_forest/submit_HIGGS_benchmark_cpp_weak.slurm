#!/bin/bash

# Directory to store the output files for the weak scaling test
OUTPUT_DIR="HIGGS_weakscaling_cpp"
mkdir -p ${OUTPUT_DIR}

# The C++ source file for the weak scaling benchmark
CPP_SOURCE_FILE="random_forest_higgs_weak.cpp"

# --- MODIFIED BLOCK START ---
# Clean up any leftover executables from previous runs to ensure a clean state.
echo "Cleaning up old executables..."
rm -f higgs_forest_weak_test_*
# --- MODIFIED BLOCK END ---

# Check if the C++ source file exists
if [ ! -f "${CPP_SOURCE_FILE}" ]; then
    echo "ERROR: C++ source file '${CPP_SOURCE_FILE}' not found."
    echo "Please ensure the C++ runner file is in the same directory as this script."
    exit 1
fi

# Define the number of threads to test with
THREADS=(1 2 4 8 16 32 64 128)
# Define the corresponding fraction of the dataset to use for each thread count
# This keeps the work-per-thread constant. 1/128, 2/128, 4/128, etc.
DATA_FRACTIONS=(0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0)

# Loop through the defined thread counts
for i in ${!THREADS[@]}; do
  NT=${THREADS[$i]}
  DATA_FRACTION=${DATA_FRACTIONS[$i]}

  # Submit a new SLURM job for each configuration
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=higgs_weak_cpp_${NT}
#SBATCH --output=${OUTPUT_DIR}/higgs_weak_cpp_${NT}.out
#SBATCH --error=${OUTPUT_DIR}/higgs_weak_cpp_${NT}.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NT}
#SBATCH --mem=32G

# Purge and load necessary modules
module purge
module load gcc/11.2
module load aocl
module load likwid

# Set environment variables required for compilation and execution
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OPENSSL3_ROOT=/home/wsgp73/openssl3
export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LIKWID_ROOT=/apps/developers/libraries/likwid/5.2.0/1/default
OPENSSL_LIB_DIR=\${OPENSSL3_ROOT}/lib64

EXECUTABLE_NAME="higgs_forest_weak_test_\${OMP_NUM_THREADS}"

echo "--- COMPILING C++ HIGGS WEAK SCALING TEST for \${OMP_NUM_THREADS} threads ---"
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

# Check if compilation was successful
if [ \$? -ne 0 ]; then
    echo "COMPILATION FAILED!"
    exit 1
fi

echo "--- RUNNING C++ EXECUTABLE with data fraction ${DATA_FRACTION} ---"
# Execute the compiled program, passing the data fraction as a command-line argument
./\${EXECUTABLE_NAME} ${DATA_FRACTION}
echo "--- JOB COMPLETE ---"

# Clean up the executable
rm ./\${EXECUTABLE_NAME}

EOF
done

echo "Submitted all HIGGS C++ weak scaling benchmark jobs to SLURM."
