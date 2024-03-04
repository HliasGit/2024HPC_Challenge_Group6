#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH -p cpu
#SBATCH -q test
#SBATCH --time 00:05:00
#SBATCH --account=p200301       # project account
#SBATCH --qos=default  

#Load Intel module
#module load VTune

#Check Intel version
vtune --version

#Profile a serial program with VTune
#srun -n 1 vtune -collect hotspots -result-dir r000hs_serial ./simple_program

#Profile MPI program with VTune
srun -n 10 vtune -collect hotspots -result-dir r000hs_mpi ../build/test
