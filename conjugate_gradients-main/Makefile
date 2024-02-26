ICC = icpx     # Intel C++ compiler
GCC = g++      # GNU C++ compiler
ICCMPI = mpiicpc   # Intel C++ compiler with MPI support
GCCMPI = mpicxx    # GNU C++ compiler with MPI support
CFLAGS = -O2   # optimizations and flags for Intel compiler
GPPFLAGS = -O2 -march=core-avx2  # optimizations and flags for GNU compiler
DEBUGFLAGS = -g  # flags for debugging
LIBSOPENBLAS = -lopenblas
LIBSFLEXIBLAS = -L/apps/USE/easybuild/release/2023.1/software/FlexiBLAS/3.3.1-GCC-12.3.0/lib64 -lflexiblas
LIBSMKL_GOMP = -Wl,--start-group /apps/USE/easybuild/release/2023.1/software/imkl/2023.1.0/mkl/latest/lib/pkgconfig/../../lib/intel64/libmkl_intel_ilp64.a /apps/USE/easybuild/release/2023.1/software/imkl/2023.1.0/mkl/latest/lib/pkgconfig/../../lib/intel64/libmkl_gnu_thread.a /apps/USE/easybuild/release/2023.1/software/imkl/2023.1.0/mkl/latest/lib/pkgconfig/../../lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl 
LIBSMKL_IOMP = -fiopenmp -fopenmp-targets=spir64  -fsycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lsycl -lstdc++ -lpthread -lm -ldl
LIBSMKL_IOMP2 = -DMKL_ILP64 -qmkl-ilp64=parallel

# Default rule for compilation with ICC
conjugate_icpx: src/conjugate_gradients.cpp
	$(ICC) $(CFLAGS) $^ -o conjugate_gradients $(LIBSMKL_GOMP)

# Rule for compilation with GCC
conjugate_gpp: src/conjugate_gradients.cpp
	$(GCC) $(GPPFLAGS) $^ -o conjugate_gradients $(LIBSMKL_GOMP)

# Rule for compilation with ICC with debug flags
conjugate_icpx_debug: src/conjugate_gradients.cpp
	$(ICC) $(CFLAGS) $(DEBUGFLAGS) $^ -o conjugate_gradients $(LIBSMKL_GOMP)

# Rule for compilation with GCC with debug flags
conjugate_gpp_debug: src/conjugate_gradients.cpp
	$(GCC) $(GPPFLAGS) $(DEBUGFLAGS) $^ -o conjugate_gradients $(LIBSMKL_GOMP)

# Default rule for compilation with ICC and MPI support
conjugate_icpx_mpi: src/conjugate_gradients.cpp
	$(ICCMPI) $(CFLAGS) $^ -o conjugate_gradients $(LIBSMKL_GOMP)

# Rule for compilation with GCC and MPI support
conjugate_gpp_mpi: src/conjugate_gradients.cpp
	$(GCCMPI) $(GPPFLAGS) $^ -o conjugate_gradients $(LIBSMKL_GOMP)

# Rule for compilation with ICC with debug flags and MPI support
conjugate_icpx_debug_mpi: src/conjugate_gradients.cpp
	$(ICCMPI) $(CFLAGS) $(DEBUGFLAGS) $^ -o conjugate_gradients $(LIBSMKL_GOMP)

# Rule for compilation with GCC with debug flags and MPI support
conjugate_gpp_debug_mpi: src/conjugate_gradients.cpp
	$(GCCMPI) $(GPPFLAGS) $(DEBUGFLAGS) $^ -o conjugate_gradients $(LIBSMKL_GOMP)

# Target to clean intermediate files and executables
clean:
	rm -f conjugate_gradients
