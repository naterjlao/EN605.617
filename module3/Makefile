main: assignment.cu helpers.o math.o 
	nvcc assignment.cu helpers.o math.o -o main

# WARNING - make clean must be performed prior to this build
csv: assignment.cu helpers.o math.o 
	nvcc assignment.cu helpers.o math.o -DCSV -o main

branch: branch.cu helpers.o
	nvcc branch.cu helpers.o -o branch

helpers.o: helpers.h helpers.cu
	nvcc -c helpers.cu

math.o: math.h math.cu
	nvcc -c math.cu

.PHONY: clean

clean:
	rm -rvf main
	rm -rvf *.o
	rm -rvf *.csv

a: all
c: clean
