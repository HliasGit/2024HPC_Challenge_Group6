# Please follow the steps for running the code on MeluXina

# Conjugate gradient method



## References
 - [this Wikipedia page](https://en.wikipedia.org/wiki/Conjugate_gradient_method).
 - https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api

## Description of the project

GPU version for Conjugate Gradient Method.
Inorder to test your code on MeluXina, please use the interactive node 
```
salloc -A p200301 --res gpudev -q dev -N 1 -t 00:30:00
```


To make the program work, I need to execute this command first
```
module load CUDA
```

Create a directory for the input and output files
```
mkdir io
```

To compile the program, I use
```
make cuda
```

To generate a random SPD system of 10000 equations and unknowns, use e.g.
```
make matrix MAT_SIZE=10000
```

Then in order to solve the system run:
```
make run_cuda
```
