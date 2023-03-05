//-----------------------------------------------------------------------------
/// @file assignment.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 6 Main Driver
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <chrono>
#include "helpers.cuh"
#include "transform.cuh"

// FUNCTION PROTOTYPES
__host__ void setup(const int totalThreads, const int numBlocks, const int blockSize, float * buffer);
__global__ void kernel_call_const(float *buffer);
__global__ void kernel_call_shared(float *buffer);

//-----------------------------------------------------------------------------
/// @brief 
/// @param argc 
/// @param argv 
/// @return 
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

	// Setup local buffer
	const size_t BUFFER_SIZE = totalThreads * sizeof(float);
	float *coordinates = (float *)malloc(BUFFER_SIZE * 2);
	setup(totalThreads, numBlocks, blockSize, &coordinates[0]);
	setup(totalThreads, numBlocks, blockSize, &coordinates[totalThreads]);
	

#if 0
	// Constant Memory Operation
	float *const_result;
	float *dev_const_buffer;
	const_result = (float *) malloc(BUFFER_SIZE);
	cudaMalloc(&dev_const_buffer, BUFFER_SIZE);
	cudaMemcpy(dev_const_buffer, buffer, BUFFER_SIZE, cudaMemcpyHostToDevice);
	const std::chrono::time_point<std::chrono::steady_clock> const_start = std::chrono::steady_clock::now();
	kernel_call_const<<<numBlocks,blockSize>>>(dev_const_buffer);
	const std::chrono::time_point<std::chrono::steady_clock> const_end = std::chrono::steady_clock::now();
	cudaMemcpy(const_result, dev_const_buffer, BUFFER_SIZE, cudaMemcpyDeviceToHost);
	cudaFree(dev_const_buffer);
	free(const_result);

	// Shared Memory Operation
	float *shared_result;
	float *dev_shared_buffer;
	shared_result = (float *) malloc(BUFFER_SIZE);
	cudaMalloc(&dev_shared_buffer, BUFFER_SIZE);
	cudaMemcpy(dev_shared_buffer, buffer, BUFFER_SIZE, cudaMemcpyHostToDevice);
	const std::chrono::time_point<std::chrono::steady_clock> shared_start = std::chrono::steady_clock::now();
	kernel_call_shared<<<numBlocks,blockSize>>>(dev_shared_buffer);
	const std::chrono::time_point<std::chrono::steady_clock> shared_end = std::chrono::steady_clock::now();
	cudaMemcpy(shared_result, dev_shared_buffer, BUFFER_SIZE, cudaMemcpyDeviceToHost);
	cudaFree(dev_shared_buffer);
	free(shared_result);

	printf("%d, %d\n",(const_end - const_start), (shared_end - shared_start));
#endif
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

	populate_random_floats<<<numBlocks, blockSize>>>(dev_buffer, r_state);

	cudaMemcpy(buffer, dev_buffer, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(r_state);
	cudaFree(dev_buffer);
}

// Kernel for const memory operations
__global__ void kernel_call_const(float *buffer)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
}

// Kernel for shared memory operations
__global__ void kernel_call_shared(float *buffer)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
}

