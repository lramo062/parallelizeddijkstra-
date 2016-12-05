all: cpu gpu main floyd run clean

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
	nvcc -o dijk main.cu -lm
	nvcc -o floyd floyd-warshal.cu -lm 

floyd:
	nvcc -g -c floyd-warshal.cu -lm

# CLEAN .o FILES
clean:
	rm *.o

