main: assignment.h assignment.cu helpers.o
	nvcc assignment.cu helpers.o -o main

helpers.o: helpers.h helpers.cu
	nvcc -c helpers.cu

.PHONY: clean

clean:
	rm -rvf main
	rm -rvf *.o
	rm -rvf *.csv

a: all
c: clean