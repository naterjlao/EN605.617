#include <stdio.h>

#include "helpers.cuh"
#include "helper_timer.h"

#define SKIP_CPU 0

__host__ void printSquareMatrix(const int matlength, float *matrix);
__host__ void cpu_rand_matrix(const int size, float *buffer);
__host__ void gpu_rand_matrix(const int totalThreads, const int numBlocks, const int blockSize, float *buffer);

int main(int argc, char **argv)
{
	// read command line arguments
	int matlength = (1 << 19);
	int blockSize = 256;

	if (argc >= 2)
	{
		matlength = atoi(argv[1]);
	}
	if (argc >= 3)
	{
		blockSize = atoi(argv[2]);
	}

	// Use a square matrix of matlength x matlength
	// The area of the matrix is totalThreads
	int totalThreads = 2 * matlength;
	int numBlocks = totalThreads / blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0)
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	// Setup Stop Watch
	StopWatchInterface *stopwatch = new StopWatchLinux();

	// Allocate Host Buffer
	float *host_buffer = (float *)malloc(sizeof(float) * totalThreads);

#if !(SKIP_CPU)
	// CPU - Random Number Generation
	stopwatch->reset();
	stopwatch->start();
	cpu_rand_matrix(totalThreads, host_buffer);
	stopwatch->stop();
	printf("%f,\n",stopwatch->getTime());
#endif
	//printSquareMatrix(matlength, host_buffer);

	// GPU - Random Number Generation
	stopwatch->reset();
	stopwatch->start();
	gpu_rand_matrix(totalThreads, numBlocks, blockSize, host_buffer);
	stopwatch->stop();
	printf("%f,\n",stopwatch->getTime());

	//printSquareMatrix(matlength, host_buffer);

	delete stopwatch;
	free(host_buffer);
}

__host__ void printSquareMatrix(const int matlength, float *matrix)
{
	for (int j = 0; j < matlength; j++)
	{
		for (int i = 0; i < matlength; i++)
		{
			printf("%f ", *(matrix + ((j * matlength) + i)));
		}
		printf("\n");
	}
}

//-----------------------------------------------------------------------------
/// @brief Sets up the working buffer by generating random floats using c++ std lib.
/// @param size Length of the buffer
/// @param buffer pointer to local host buffer
/// @return None; buffer is modified
//-----------------------------------------------------------------------------
__host__ void cpu_rand_matrix(const int size, float *buffer)
{
	std::srand(420);
	for (int i = 0; i < size; i++)
	{
		buffer[i] = (float) (std::rand() / ((float) RAND_MAX));
	}
}

//-----------------------------------------------------------------------------
/// @brief Sets up the working buffer by generating random floats using curand.
/// @param totalThreads Length of the buffer
/// @param numBlocks Cuda Number of Blocks
/// @param blockSize Cuda Block Size
/// @param buffer pointer to local host buffer
/// @return None; buffer is modified
//-----------------------------------------------------------------------------
__host__ void gpu_rand_matrix(const int totalThreads, const int numBlocks, const int blockSize, float *buffer)
{
	curandState *r_state;
	cudaMalloc(&r_state, totalThreads * sizeof(curandState));
	setup_random<<<numBlocks, blockSize>>>(r_state);

	float *dev_buffer;
	cudaMalloc(&dev_buffer, totalThreads * sizeof(float));

	// Create random F32s
	populate_random_floats<<<numBlocks, blockSize>>>(dev_buffer, r_state);
	cudaDeviceSynchronize();

	cudaMemcpy(buffer, dev_buffer, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(r_state);
	cudaFree(dev_buffer);
}