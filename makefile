all: cpu main run clean

# CPU
cpu: cpu.c
	gcc -g -Wall -c  cpu.c

# MAIN
main: main.c
	gcc -g -Wall -c  main.c

run:
	gcc -o main main.o

clean:
	rm *.o

