#!/bin/bash
#SBATCH -J linaro-forge
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --time=00:15:00
#SBATCH --account=p200301
#SBATCH --partition=cpu
#SBATCH --qos=default

module purge
module load Linaro-Forge/23.0.3-GCC-12.3.0 intel/
#make conjugate_icpx_debug_mpi 
map --profile srun ./conjugate_gradients io/matrix.bin io/rhs.bin io/sol.bin