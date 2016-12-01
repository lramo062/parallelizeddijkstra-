all: cpu gpu main run clean

# CPU
cpu: cpu.c
	gcc -g -Wall -c  cpu.c

# GPU
gpu: gpu.c
	gcc -g -Wall -c gpu.c

# MAIN
main: main.c
	gcc -g -Wall -c  main.c

# CREATE EXECUTABLE
run:
	gcc -o main main.c

# CLEAN .o FILES
clean:
	rm *.o

