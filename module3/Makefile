all: assignment.cu helpers.o math.o 
	nvcc assignment.cu helpers.o math.o -o assignment

helpers.o: helpers.h helpers.cu
	nvcc -c helpers.cu

math.o: math.h math.cu
	nvcc -c math.cu

.PHONY: clean

clean:
	rm -rvf assignment
	rm -rvf *.o

a: all
c: clean
