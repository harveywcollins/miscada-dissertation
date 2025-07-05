#!/bin/bash

for NT in 32; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=higgs_params_${NT}
#SBATCH --output=HIGGS_params_${NT}.out
#SBATCH --error=HIGGS_params_${NT}.err
#SBATCH --partition=shared
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NT}
#SBATCH --mem=32G
#SBATCH --hint=nomultithread

module purge
module load gcc/11.2
module load aocl

export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LD_LIBRARY_PATH=\$AOCLDA_ROOT/lib/LP64:\$AOCL_ROOT/lib_LP64:\$LD_LIBRARY_PATH
export PYTHONPATH=\$AOCLDA_ROOT/python_package:\$PYTHONPATH

export AOCLDA_BIND_THREADS=TRUE
export OMP_NUM_THREADS=${NT}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

source /home/wsgp73/miscada-dissertation/code/myenv312/bin/activate

python3 decision_forest_HIGGS_params.py
EOF
done
