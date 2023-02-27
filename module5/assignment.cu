//-----------------------------------------------------------------------------
/// @file assignment.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 5 Main Driver
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <chrono>
#include "assignment.h"
#include "helpers.h"

// This does absolutely nothing
__constant__ float const_addr[NUM_ELEMENTS] = {0.0, 0.0, 0.0, 0.0};
__constant__ float const_subb[NUM_ELEMENTS] = {0.0, 0.0, 0.0, 0.0};
__constant__ float const_mult[NUM_ELEMENTS] = {1.0, 1.0, 1.0, 1.0};
__constant__ float const_divd[NUM_ELEMENTS] = {1.0, 1.0, 1.0, 1.0};

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
	float *buffer = (float *)malloc(BUFFER_SIZE);
	setup(totalThreads, numBlocks, blockSize, buffer);

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

	printf("%d, %d\n",(const_end - const_start), (shared_end - shared_start));

	// Cleanup
	free(buffer);
	free(const_result);
	free(shared_result);
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
	buffer[thread_idx] = operation(buffer[thread_idx], const_addr, const_subb, const_mult, const_divd);
}

// Kernel for shared memory operations
__global__ void kernel_call_shared(float *buffer)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Setup shared memory, same contents as constant memory
	__shared__ float shared_addr[NUM_ELEMENTS];
	__shared__ float shared_subb[NUM_ELEMENTS];
	__shared__ float shared_mult[NUM_ELEMENTS];
	__shared__ float shared_divd[NUM_ELEMENTS];
	shared_addr[0] = 0.0;
	shared_addr[1] = 0.0;
	shared_addr[2] = 0.0;
	shared_addr[3] = 0.0;
	shared_subb[0] = 0.0;
	shared_subb[1] = 0.0;
	shared_subb[2] = 0.0;
	shared_subb[3] = 0.0;
	shared_mult[0] = 1.0;
	shared_mult[1] = 1.0;
	shared_mult[2] = 1.0;
	shared_mult[3] = 1.0;
	shared_divd[0] = 1.0;
	shared_divd[1] = 1.0;
	shared_divd[2] = 1.0;
	shared_divd[3] = 1.0;

	// Perform operation
	__syncthreads();
	buffer[thread_idx] = operation(buffer[thread_idx], shared_addr, shared_subb, shared_mult, shared_divd);
}

// Iterates through a memory array and operate over the input value.
__device__ float operation(float val,
						  float arr_addr[NUM_ELEMENTS],
						  float arr_subb[NUM_ELEMENTS],
						  float arr_mult[NUM_ELEMENTS],
						  float arr_divd[NUM_ELEMENTS])
{
	float retval = val;
	for (int i = 0; i < NUM_ELEMENTS; i++)
	{
		retval = math(retval, arr_addr[i], arr_subb[i], arr_mult[i], arr_divd[i]);
	}
	return retval;
}

// This performs four basic math operations
__device__ float math(const float i,
					  const float add,
					  const float sub,
					  const float mul,
					  const float div)
{
	float retval = i;
	retval = retval + add;
	retval = retval - sub;
	retval = retval * mul;
	retval = retval / div;
	return retval;
}
