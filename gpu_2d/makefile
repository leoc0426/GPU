CUDA_LIB := /usr/local/cuda/lib64 -lcuda -lcudart

all: CPU GPU
	g++ main.o gpumain.o -o3 -o test.run -L $(CUDA_LIB)

CPU:
	g++ main.c -c
GPU:
	nvcc gpumain.cu -c

clear:
	rm *.o
