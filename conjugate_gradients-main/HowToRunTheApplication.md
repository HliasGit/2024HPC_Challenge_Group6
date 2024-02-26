To test the application:

1. salloc -A p200301 --res cpudev -q dev -N 1 -t 01:00:00 -p cpu
2. cd /project/home/p200301/group6/2024-EUMaster4HPC-Student-Challenge/conjugate_gradients-main
3. source modulesBeforeCompiling.sh 
4. make conjugate_icpx_mpi
5. srun -N 1 -n 1 -p cpu random_spd_system.sh 10000 io/matrix.bin io/rhs.bin
6. srun -N 1 -n 1 -p cpu conjugate_gradients io/matrix.bin io/rhs.bin io/sol.bin