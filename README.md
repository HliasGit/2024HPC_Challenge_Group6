# 2024HPC_Challenge_Group6
The following branch aims to demonstrate how the serial problem exhibits a substantial margin for improvement in terms of performance through parallel computation.To parallelize the code, we employed OpenMP, achieving excellent results in a versy simply manner.

## Compile the code
Inorder to test your code on MeluXina, please use the interactive node:

``` salloc -A p200301 --res cpudev -q dev -N 1 -t 00:30:00 ```

Compiling using g++ compiler

``` gcc -o helloworld_omp helloworld_OMP.c -fopenmp ```
