//-----------------------------------------------------------------------------
/// @file assignment.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 6 Main Driver
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <chrono>
#include <math.h>
#include "helpers.cuh"
#include "transform.cuh"

#define PRINT_RESULTS 0
typedef void (*KERNEL_FUNCTION)(float *, size_t);
__host__ void setup(const int totalThreads, const int numBlocks, const int blockSize, float *buffer);
__host__ std::chrono::duration<int64_t, std::nano> run_test(const int totalThreads, const int numBlocks, const int blockSize, float *buffer, const size_t buffer_size, KERNEL_FUNCTION kernel_function);

//-----------------------------------------------------------------------------
/// @brief Main Driver
//-----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;

	if (argc >= 2)
	{
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3)
	{
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads / blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0)
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	// Setup 2D matrix, note that each thread operates on a coordinate
	// A coordinate is comprised of two points and the buffer datastructure
	// is modeled in an float[2][n_points] array.
	// Access to the coordinate at idx should be as follows:
	// x <- coordinates[idx]
	// y <- coordinates[idx + n_points]
	const size_t BUFFER_SIZE = 2 * totalThreads * sizeof(float);
	float *coordinates = (float *)malloc(BUFFER_SIZE);

	// Allocate coordinates, note that this is called twice for the x and y elements
	setup(totalThreads, numBlocks, blockSize, &coordinates[0]);
	setup(totalThreads, numBlocks, blockSize, &coordinates[totalThreads]);

	// Execute register and global time tests
	printf("%d,", run_test(totalThreads, numBlocks, blockSize, coordinates, BUFFER_SIZE, kernel_call_global));
	printf("%d\n", run_test(totalThreads, numBlocks, blockSize, coordinates, BUFFER_SIZE, kernel_call_register));

	// Cleanup
	free(coordinates);
}

//-----------------------------------------------------------------------------
/// @brief Sets up the working buffer by generating random floats
/// @param totalThreads Length of the buffer
/// @param numBlocks Cuda Number of Blocks
/// @param blockSize Cuda Block Size
/// @param buffer pointer to local host buffer
/// @return None; buffer is modified
//-----------------------------------------------------------------------------
__host__ void setup(const int totalThreads, const int numBlocks, const int blockSize, float *buffer)
{
	curandState *r_state;
	cudaMalloc(&r_state, totalThreads * sizeof(curandState));
	setup_random<<<numBlocks, blockSize>>>(r_state);

	float *dev_buffer;
	cudaMalloc(&dev_buffer, totalThreads * sizeof(float));

	// Create random F32s
	populate_random_floats<<<numBlocks, blockSize>>>(dev_buffer, r_state);

	cudaMemcpy(buffer, dev_buffer, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(r_state);
	cudaFree(dev_buffer);
}

//-----------------------------------------------------------------------------
/// @brief Executes a CUDA kernel call timed test
/// @param totalThreads Number of CUDA Threads
/// @param numBlocks Number of CUDA Blocks
/// @param blockSize CUDA Block Size
/// @param buffer Pointer to host buffer
/// @param buffer_size Size of the host buffer in bytes
/// @param kernel_function Pointer to a KERNEL_FUNCTION
/// @return The duration of time to execute the kernel function
//-----------------------------------------------------------------------------
__host__ std::chrono::duration<int64_t, std::nano> run_test(
	const int totalThreads, const int numBlocks, const int blockSize,
	float *buffer, const size_t buffer_size, KERNEL_FUNCTION kernel_function)
{
#if PRINT_RESULTS
	float *result = (float *)malloc(buffer_size);
#endif
	// Allocate device buffer
	float *dev_buffer;
	cudaMalloc(&dev_buffer, buffer_size);
	cudaMemcpy(dev_buffer, buffer, buffer_size, cudaMemcpyHostToDevice);

	// Execute the kernel function
	const std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
	kernel_function<<<numBlocks, blockSize>>>(dev_buffer, totalThreads);
	cudaDeviceSynchronize();
	const std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();

#if PRINT_RESULTS
	cudaMemcpy(result, dev_buffer, buffer_size, cudaMemcpyDeviceToHost);
	for (size_t idx = 0; idx < totalThreads; idx++)
		printf("(%f,%f)\n", result[idx], result[idx + totalThreads]);
	free(result);
#endif
	cudaFree(dev_buffer);
	return end - start;
}
