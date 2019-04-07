SM := 35

CC := gcc
MPICC := mpicc
NVCC := nvcc

CFLAGS = -std=c99
NVCCFLAGS = -O3

GENCODE_FLAGS = -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS = -lcudadevrt -lcudart

BUILDDIR = build

TARGET = kmeans
all: $(TARGET)
mpi: $(TARGET)-mpi
cuda: $(TARGET)-cuda

$(TARGET): $(BUILDDIR)/dlink.o $(BUILDDIR)/txtproc.o $(BUILDDIR)/cnfparser.o $(BUILDDIR)/main.o $(BUILDDIR)/$(TARGET).o
	$(MPICC) $(CFLAGS) $^ -o $@ $(LIB_FLAGS)

$(TARGET)-mpi: $(BUILDDIR)/txtproc.o $(BUILDDIR)/cnfparser.o $(BUILDDIR)/mpi_main.o
	$(MPICC) $(CFLAGS) $^ -o $@ 

$(TARGET)-cuda: $(BUILDDIR)/dlink_cuda.o $(BUILDDIR)/txtproc.o $(BUILDDIR)/cnfparser.o $(BUILDDIR)/cuda_main.o $(BUILDDIR)/$(TARGET)_cuda.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_FLAGS)

$(BUILDDIR)/dlink.o: $(BUILDDIR)/$(TARGET).o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

$(BUILDDIR)/dlink_cuda.o: $(BUILDDIR)/$(TARGET)_cuda.o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

$(BUILDDIR)/main.o: src/mpi_cuda/main.c
	$(MPICC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/mpi_main.o: src/mpi/main.c
	$(MPICC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/cuda_main.o: src/cuda/main.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/txtproc.o: src/preprocess/txtproc.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/cnfparser.o: src/preprocess/cnfparser.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/$(TARGET).o: cuda/mpi_cuda/$(TARGET)_kernal.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 

$(BUILDDIR)/$(TARGET)_cuda.o: cuda/cuda/$(TARGET)_kernal.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 
clean:
	rm -f $(BUILDDIR)/*.o $(TARGET) $(TARGET)-mpi $(TARGET)-cuda

