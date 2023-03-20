#include <stdio.h>

#include "helpers.cuh"
#include "helper_timer.h"

#define SKIP_CPU 0

__host__ void printSquareMatrix(const int matlength, float *matrix);
__host__ void identityMatrix(const int matlength, float *matrix);
__host__ float cpu_rand_matrix(const size_t size,
	float *buffer);
__host__ float gpu_rand_matrix(const int totalThreads,
	const int numBlocks,
	const int blockSize,
	float *buffer);

int main(int argc, char **argv)
{
	// read command line arguments
	int matlength = (1 << 10);
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
	int totalThreads = matlength * matlength;
	int numBlocks = totalThreads / blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0)
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	// Allocate Matrices
	float *matrix_buffer = (float *)malloc(sizeof(float) * totalThreads);
	float *matrix_identy = (float *)malloc(sizeof(float) * totalThreads);

	printf("%d, ", totalThreads);

#if !(SKIP_CPU)
	// CPU - Random Number Generation
	printf("%f, ",cpu_rand_matrix(totalThreads, matrix_buffer));
#endif

	// GPU - Random Number Generation
	printf("%f,\n",gpu_rand_matrix(totalThreads, numBlocks, blockSize, matrix_buffer));

	//identityMatrix(matlength, matrix_identy);
	//printSquareMatrix(matlength, matrix_buffer);

	free(matrix_buffer);
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

__host__ void identityMatrix(const int matlength, float *matrix)
{
	for (size_t d = 0; d < matlength; d++)
	{
		*(matrix + ((d * matlength) + d)) = 1.0;
	}
}

//-----------------------------------------------------------------------------
/// @brief Sets up the working buffer by generating random floats using c++ std lib.
/// @param size Length of the buffer
/// @param buffer pointer to local host buffer
/// @return None; buffer is modified
//-----------------------------------------------------------------------------
__host__ float cpu_rand_matrix(const size_t size,
	float *buffer)
{
	StopWatchInterface *stopwatch = new StopWatchLinux();
	srand(420);

	stopwatch->reset();
	stopwatch->start();
	for (size_t i = 0; i < size; i++)
	{
		buffer[i] = (float) (rand() / ((float) RAND_MAX));
	}
	stopwatch->stop();
	
	// Measure time - note that this is measured in the CPU and is in milliseconds
	float timer = stopwatch->getTime();
	delete stopwatch;
	return timer;
}

//-----------------------------------------------------------------------------
/// @brief Sets up the working buffer by generating random floats using curand.
/// @param totalThreads Length of the buffer
/// @param numBlocks Cuda Number of Blocks
/// @param blockSize Cuda Block Size
/// @param buffer pointer to local host buffer
/// @return None; buffer is modified
//-----------------------------------------------------------------------------
__host__ float gpu_rand_matrix(const int totalThreads,
	const int numBlocks,
	const int blockSize,
	float *buffer)
{
	curandState *r_state;
	cudaMalloc(&r_state, totalThreads * sizeof(curandState));
	setup_random<<<numBlocks, blockSize>>>(r_state);

	float *dev_buffer;
	cudaMalloc(&dev_buffer, totalThreads * sizeof(float));

	// Setup Timer
	cudaEvent_t start;
	cudaEvent_t end;
	float timer;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Create random F32s
	cudaEventRecord(start);
	populate_random_floats<<<numBlocks, blockSize>>>(dev_buffer, r_state);
	cudaDeviceSynchronize();
	cudaEventRecord(end);
	
	// Copy To Host
	cudaMemcpy(buffer, dev_buffer, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(r_state);
	cudaFree(dev_buffer);
	
	// Measure Time - note that this measured from CUDA and is in milliseconds
	cudaEventElapsedTime(&timer, start, end);
	return timer;
}