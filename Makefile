SM := 35

CC := mpicxx
NVCC := nvcc

CFLAGS = -std=c++11
NVCCFLAGS = -O3

GENCODE_FLAGS = -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS = -lcudadevrt -lcudart

BUILDDIR = build

TARGET = kmeans
all: $(TARGET)
#%.o: %.c
#	$(CC) -c -o $@ $^ $(CFLAGS)

#$(TARGET): $(TARGET).o
#	$(CC) -o $@ $^ $(CFLAGS)

$(TARGET): $(BUILDDIR)/dlink.o $(BUILDDIR)/txtproc.o $(BUILDDIR)/cnfparser.o $(BUILDDIR)/main.o $(BUILDDIR)/$(TARGET).o
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_FLAGS)

$(BUILDDIR)/dlink.o: $(BUILDDIR)/$(TARGET).o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

$(BUILDDIR)/main.o: src/main.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/txtproc.o: src/txtproc.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/cnfparser.o: src/cnfparser.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/$(TARGET).o: cuda/$(TARGET)_kernal.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 

clean:
	rm -f $(BUILDDIR)/*.o $(TARGET)

