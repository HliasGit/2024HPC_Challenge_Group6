# Defines compiler
CC=nvcc

SERIAL_PROGRAM = "serial_cg"
CUDA_PROGRAM = "test"
# These are the options we pass to the compiler.
# -std=c++14 means we want to use the C++14 standard.
# -stdlib=libc++ specifies that we want to use the standard library implementation called libc++
# -g specifies that we want to include "debugging symbols" which allows us to use a debugging program.
# -O0 specifies to do no optimizations on our code.
# -Wall, -Wextra, and -pedantic tells the compiler to look out for common problems with our code. -Werror makes it so that these warnings stop compilation.
CFLAGS = -std=c++20 -c -O0 

# Additional flags for CUDA
CUDAFLAGS = -arch=sm_35

# Matrix size
MAT_SIZE = 10000

serial:
	$(CC) src/$(SERIAL_PROGRAM).cpp -o build/$(SERIAL_PROGRAM)

cuda:
	$(CC) -lcublas src/$(CUDA_PROGRAM).cu -o build/$(CUDA_PROGRAM)

clean:
	rm -f build/test
	rm -f build/serial_cg
        
run_serial:
	build/./$(SERIAL_PROGRAM) io/matrix.bin io/rhs.bin
run_cuda:
	build/./$(CUDA_PROGRAM) io/matrix.bin io/rhs.bin

