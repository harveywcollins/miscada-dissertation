#!/bin/bash

for NT in 1 2 3 4 5 6 7 8 12 16 24 32; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=mnist_strong_${NT}
#SBATCH --output=MNIST_strongscaling/mnist_strong_${NT}.out
#SBATCH --error=MNIST_strongscaling/mnist_strong_${NT}.err
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NT}
#SBATCH --mem=32G

module purge
module load gcc/11.2
module load aocl

export AOCL_ROOT=/home/wsgp73/aocl/5.0.0/gcc
export AOCLDA_ROOT=/home/wsgp73/miscada-dissertation/code/aocl-data-analytics/build/install
export LD_LIBRARY_PATH=\$AOCLDA_ROOT/lib/LP64:\$AOCL_ROOT/lib_LP64:/apps/developers/compilers/gcc/11.2/1/default/lib64:\$LD_LIBRARY_PATH
export PYTHONPATH=\$AOCLDA_ROOT/python_package:\$PYTHONPATH

export AOCLDA_BIND_THREADS=0
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

source /home/wsgp73/miscada-dissertation/code/myenv312/bin/activate
python3 decision_forest_MNIST.py
EOF
done

