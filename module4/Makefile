.PHONY: all clean

all: pageable pinned

pageable: assignment.cu cypher.o fileio_pageable.o
	nvcc assignment.cu cypher.o fileio_pageable.o -o pageable

pinned: assignment.cu cypher.o fileio_pinned.o
	nvcc assignment.cu cypher.o fileio_pinned.o -o pinned

cypher.o: cypher.h cypher.cu
	nvcc -c cypher.cu

fileio_pageable.o: fileio.h fileio.cpp
	g++ -c fileio.cpp -o fileio_pageable.o

fileio_pinned.o: fileio.h fileio.cu
	nvcc -c fileio.cu -o fileio_pinned.o

fileio.cpp:
	cp fileio.cu fileio.cpp

clean:
	rm -rvf pageable
	rm -rvf pinned
	rm -rvf fileio.cpp
	rm -rvf *.o
	rm -rvf *.txt

a: all
c: clean