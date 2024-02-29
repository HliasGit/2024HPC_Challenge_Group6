# 2024HPC_Challenge_Group6

## How to run the example
First you need to enter a dev node.

Once you're in, you need to compile the project. To do it you want to load the necessary modules.

```
module load OpenMPI
module load CMake
```

### Create the matrix and the rhs
To do it follow the example provided by the organizer of the challenge.

### Compile and run
Run the bash script called "prova.sh"

```
sbatch script/prova.sh
```