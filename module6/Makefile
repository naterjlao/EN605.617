main: assignment.cu helpers.o transform.o
	nvcc assignment.cu helpers.o transform.o -o main

helpers.o: helpers.cuh helpers.cu
	nvcc -c helpers.cu

transform.o: transform.cuh transform.cu
	nvcc -c transform.cu

.PHONY: clean

clean:
	rm -rvf main
	rm -rvf *.o
	rm -rvf *.csv

a: all
c: clean