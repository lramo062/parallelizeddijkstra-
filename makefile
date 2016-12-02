all: cpu gpu main run clean

# CPU
cpu: cpu.cu
	nvcc -g -c  cpu.cu -lm

# GPU
gpu: gpu.cu
	nvcc -g -c gpu.cu -lm

# MAIN
main: main.cu
	nvcc -g -c  main.cu -lm 

# CREATE EXECUTABLE
run:
	nvcc -o main main.cu -lm

# CLEAN .o FILES
clean:
	rm *.o

