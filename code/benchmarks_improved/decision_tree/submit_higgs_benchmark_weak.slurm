#!/bin/bash

for NT in 1 2 4 8 16 32 64 128; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=higgs_weak_${NT}
#SBATCH --output=HIGGS_weakscaling/higgs_weak_${NT}.out
#SBATCH --error=HIGGS_weakscaling/higgs_weak_${NT}.err
#SBATCH --partition=shared
#SBATCH --time=23:30:00
#SBATCH -c ${NT}
#SBATCH --exclusive

#SBATCH --mem-per-cpu=1G 

module purge
module load gcc/11.2
module load aocl

export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LD_LIBRARY_PATH=\$AOCLDA_ROOT/lib/ILP64:\$AOCLDA_ROOT/lib/LP64:\$AOCL_ROOT/lib_LP64:\$LD_LIBRARY_PATH
export PYTHONPATH=\$AOCLDA_ROOT/python_package:\$PYTHONPATH

export LD_PRELOAD=/home/wsgp73/miscada-dissertation/code/benchmarks_baseline/noaffinity.so

export OMP_NUM_THREADS=${NT}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export KMP_AFFINITY=verbose,granularity=fine,scatter

source /home/wsgp73/miscada-dissertation/code/myenv312/bin/activate

python3 decision_forest_HIGGS_weak.py
EOF
done
