#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
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
srun build/test io/matrix.bin io/rhs.bin io/sol.bin