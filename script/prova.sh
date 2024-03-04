#!/bin/bash -l
#SBATCH --nodes 2    #nodes
#SBATCH --ntasks 30     #MPI processes
#SBATCH --cpus-per-task=16 #OpenMP cores
#SBATCH -p cpu
#SBATCH -q test
#SBATCH --time 00:30:00
#SBATCH --account=p200301       # project account
#SBATCH --qos=default           # SLURM qos

srun conjugate_gradients-main/conjugate_gradients io/matrix.bin io/rhs.bin