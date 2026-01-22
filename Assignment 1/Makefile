
NVCC    = nvcc
OBJ     = main.o kernel.o
EXE     = daxpy


default: $(EXE)

%.o: %.cu
	$(NVCC) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

