

pageable: assignment.cu cypher.o fileio_pageable.o
	nvcc assignment.cu cypher.o fileio_pageable.o -o pageable

pinned: assignment.cu cypher.o fileio_pinned.o
	nvcc assignment.cu cypher.o fileio_pinned.o -o pageable

cypher.o: cypher.h cypher.cu
	nvcc -c cypher.cu

fileio_pageable.o: fileio.h fileio_pageable.cpp
	g++ -c fileio_pageable.cpp

.PHONY: clean

clean:
	rm -rvf pageable
	rm -rvf pinned
	rm -rvf *.o

a: all
c: clean