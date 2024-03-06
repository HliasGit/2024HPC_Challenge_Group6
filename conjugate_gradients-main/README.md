# Please follow the steps for running the code on MeluXina

## References
 - [this Wikipedia page](https://en.wikipedia.org/wiki/Conjugate_gradient_method).
 - https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api

## Description of the project

The project involves implementing a GPU version of the Conjugate Gradient Method. This adaptation leverages the computational power and parallel processing capabilities of Graphics Processing Units (GPUs) to enhance the efficiency and speed of the Conjugate Gradient Method, a popular iterative numerical technique used for solving linear systems of equations. By harnessing the parallelization capabilities offered by GPU computing, the goal is to optimize the performance of the Conjugate Gradient Method, allowing for faster and more efficient solutions to large-scale linear algebra problems.

Inorder to test your code on MeluXina, please use the interactive node 
```
salloc -A p200301 --res gpudev -q dev -N 1 -t 00:30:00
```


To make the program work, execute this command first
```
module load CUDA
```

Create a directory for the input and output files
```
mkdir io
```

To compile the program use
```
make cuda
```

To generate a random SPD system of 10000 equations and unknowns, use e.g.. The default size is 10000.
```
make matrix MAT_SIZE=10000
```

Then in order to solve the system run:
```
make run_cuda
```
