# 2024HPC_Challenge_Group6
The following branch aims to demonstrate how the serial problem exhibits a substantial margin for improvement in terms of performance through parallel computation.To parallelize the code, we employed OpenMP, achieving excellent results in a versy simply manner.

## Compile the code
Inorder to test your code on MeluXina, please use the interactive node:

```
salloc -A p200301 --res cpudev -q dev -N 1 -t 00:30:00
```
Create a directory for the input and output files
```
mkdir io
```
Compiling using g++ compiler

```
g++ -o conjugate_gradients conjugate_gradients.cpp -fopenmp 
```
Set number of threads using
```
export OMP_NUM_THREADS=8
```
To generate a random SPD system of 10000 equations and unknowns, use e.g.
```
./random_spd_system.sh 10000 io/matrix.bin io/rhs.bin
```
This program measures the execution time of the conjugate_gradients function using a 1000 x 1000 matrix. We conducted the analysis with an increasing number of threads, and to obtain a reasonably accurate measurement, we performed four measurements for each configuration. 

Please refer to the notebook (```scripts/plot_speed_up.ipynb```) to view the results.
