main: assignment.c cypher.o fileio.o
	nvcc assignment.c cypher.o fileio.o -o main

cypher.o: cypher.h cypher.cu
	nvcc -c cypher.cu

fileio.o: fileio.h fileio.c
	nvcc -c fileio.c

.PHONY: clean

clean:
	rm -rvf main
	rm -rvf *.o

a: all
c: clean