main: assignment.cu helpers.o helper_timer.h exception.h
	nvcc assignment.cu helpers.o -lcublas -o main

helpers.o: helpers.cuh helpers.cu
	nvcc -c helpers.cu

.PHONY: clean

clean:
	rm -rvf main
	rm -rvf *.o
	rm -rvf *.csv

a: all
c: clean