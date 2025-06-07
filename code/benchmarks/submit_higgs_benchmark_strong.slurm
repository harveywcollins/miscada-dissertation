#!/bin/bash

for NT in 2 4 8 16; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=higgs_strong_${NT}
#SBATCH --output=HIGGS_strongscaling/higgs_strong_${NT}.out
#SBATCH --error=HIGGS_strongscaling/higgs_strong_${NT}.err
#SBATCH --partition=shared
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NT}
#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl

export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install

export LD_LIBRARY_PATH=\$AOCLDA_ROOT/lib/LP64:/home/wsgp73/aocl/5.0.0/gcc/lib_LP64:/apps/developers/compilers/gcc/11.2/1/default/lib64:\$LD_LIBRARY_PATH

export PYTHONPATH=\$AOCLDA_ROOT/python_package:\$PYTHONPATH

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export AOCLDA_BIND_THREADS=0

source /home/wsgp73/miscada-dissertation/code/myenv312/bin/activate

export OMP_NUM_THREADS=${NT}

python3 decision_forest_HIGGS.py
EOF
done
