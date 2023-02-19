#include <stdio.h>
#include "fileio.h"
#include "cypher.h"


int main(int argc, char** argv)
{
	// Read arguments
	if (argc < 4)
	{
		fprintf(stderr,"ERROR: invalid number of arguments, expecting:\n");
		fprintf(stderr,"main <INPUT_FILE> <CYPHER_SHIFT> <OUTPUT_FILE>\n");
		return -1;
	}
	const char *input_file = argv[1];
	const int cypher_shift = atoi(argv[2]);
	const char *output_file = argv[3];

	// Read in file and store in Host buffer
	char *buffer;
	size_t length = read_file(input_file, &buffer);
	if (length == 0)
	{
		fprintf(stderr,"ERROR: empty file or file open failed\n");
		return -1;
	}

#if 0
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
#else
	for (size_t i = 0; i < length; i++)
		buffer[i] = buffer[i] + cypher_shift;
#endif

	// Output the conversion to stdout
	//printf("%s\n",buffer);
	write_file(output_file, buffer, length);

	// Free Host Memory
	if (buffer != 0)
	{
		free(buffer);
		buffer = 0;
	}

	return 0;
}
