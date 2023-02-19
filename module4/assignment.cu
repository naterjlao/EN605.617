//-----------------------------------------------------------------------------
/// @file assignment.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 4 Main Driver
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <chrono>
#include "fileio.h"
#include "cypher.h"

//-----------------------------------------------------------------------------
/// @brief Main Driver
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
	// Read arguments
	if (argc < 3)
	{
		fprintf(stderr,"ERROR: invalid number of arguments, expecting:\n");
		fprintf(stderr,"main <INPUT_FILE> <CYPHER_SHIFT> [OUTPUT_FILE]\n");
		return -1;
	}
	const char *input_file = argv[1];
	const int cypher_shift = atoi(argv[2]);
	const char *output_file = 0;
	if (argc > 3)
		output_file = argv[3];

	// Read in file and store in Host buffer
	/// @note Interestingly (or not surprisingly since it is non-pageable),
	/// pinned memory takes up more time to read
	char *buffer;
	size_t length = read_file(input_file, &buffer);
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

	// Start timer
	const std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();

	// Copy to Device Memory
	cudaMemcpy(device_buffer, buffer, length, cudaMemcpyHostToDevice);

	// Run the conversion on the GPU
	caesar_cypher<<<NUM_BLOCKS, BLOCK_SIZE>>>(device_buffer, cypher_shift);

	// Copy to Host Memory
	cudaMemcpy(buffer, device_buffer, length, cudaMemcpyDeviceToHost);
	
	// Stop timer
	const std::chrono::time_point<std::chrono::steady_clock> stop_time = std::chrono::steady_clock::now();
	printf("%u\n", (stop_time - start_time));

	// Free Device Memory
	cudaFree(device_buffer);
	device_buffer = 0;

	// Output results to file
	/// @note This takes up the majority of the time, so this is optional
	if (output_file != 0)
		write_file(output_file, buffer, length);

	// Free Host Memory
	free_buffer(&buffer);

	return 0;
}
