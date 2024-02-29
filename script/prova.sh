#!/bin/bash -l
#SBATCH -N 3
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=16
#SBATCH -p cpu
#SBATCH -q test
#SBATCH --time 00:30:00
#SBATCH --account=p200301       # project account
#SBATCH --qos=default           # SLURM qos

if [ ! -d "build" ]; then
  echo "build does not exist."
  mkdir build
fi

cd build
cmake ..
make

cd ..

#Execute the program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun  --mpi=pspmix --cpus-per-task=$SLURM_CPUS_PER_TASK build/test io/matrix.bin io/rhs.bin