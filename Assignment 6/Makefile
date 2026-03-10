
NVCC        = nvcc
NVCC_FLAGS  = -O3 -arch=sm_70 -std=c++11
OBJ         = main.o kernel.o
EXE         = reduction


default: $(EXE)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

