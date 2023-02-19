main: assignment.c cypher.o
	nvcc assignment.c cypher.o -o main

cypher.o: cypher.h cypher.cu
	nvcc -c cypher.cu

.PHONY: clean

clean:
	rm -rvf main
	rm -rvf *.o

a: all
c: clean