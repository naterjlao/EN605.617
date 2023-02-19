#include <stdio.h>
#include "fileio.h"
#include "cypher.h"


int main(int argc, char** argv)
{
	// Read arguments
	if (argc < 3)
	{
		fprintf(stderr,"ERROR: invalid number of arguments, expecting:\n");
		fprintf(stderr,"main <FILENAME> <CYPHER_SHIFT>\n");
		return -1;
	}
	const char *filename = argv[1];
	const int cypher_shift = atoi(argv[2]);

	// Read in file and store in Host buffer
	char *buffer;
	size_t length = read_file(filename, &buffer);
	if (length == 0)
	{
		fprintf(stderr,"ERROR: empty file or file open failed\n");
		return -1;
	}

	// Setup Device Memory
	const int BLOCK_SIZE = 1024;
	const int NUM_BLOCKS = length / BLOCK_SIZE + ((length % BLOCK_SIZE > 0) ? 1 : 0);
	char *device_buffer;
	cudaMalloc(&device_buffer, (NUM_BLOCKS * BLOCK_SIZE) * sizeof(char)); // We allocate more if needed
	cudaMemcpy(device_buffer, buffer, length, cudaMemcpyHostToDevice);

	// Run the conversion on the GPU
	caesar_cypher<<<NUM_BLOCKS, BLOCK_SIZE>>>(device_buffer, cypher_shift);

	// Free Device Memory
	cudaMemcpy(buffer, device_buffer, length, cudaMemcpyDeviceToHost);
	cudaFree(device_buffer);
	device_buffer = 0;

	// Output the conversion to stdout
	printf("%s\n",buffer);

	// Free Host Memory
	if (buffer != 0)
	{
		free(buffer);
		buffer = 0;
	}

	return 0;
}
