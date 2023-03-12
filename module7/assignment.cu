//-----------------------------------------------------------------------------
/// @file assignment.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 6 Main Driver
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include "helpers.cuh"
#include "transform.cuh"

#define PRINT_RESULTS 0
typedef void (*KERNEL_FUNCTION)(float *, size_t);
__host__ void setup(
	const int totalThreads,
	const int numBlocks,
	const int blockSize,
	float *buffer);
__host__ float run_serial(
	const int totalThreads,
	const int numBlocks,
	const int
		blockSize,
	float *buffer,
	const size_t buffer_size,
	KERNEL_FUNCTION kernel_function,
	size_t iterations);
__host__ float run_async(
	const int totalThreads,
	const int numBlocks,
	const int
		blockSize,
	float *buffer,
	const size_t buffer_size,
	KERNEL_FUNCTION kernel_function,
	size_t iterations);

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

#if PRINT_RESULTS
	for (size_t i = 0; i < totalThreads; i++)
		printf("(%f, %f)\n", coordinates[i], coordinates[i + totalThreads]);
#endif

	// Execute serial and async time tests
	const size_t ITERATIONS = 2048;
	printf("%f,", run_serial(totalThreads, numBlocks, blockSize, coordinates, BUFFER_SIZE, kernel_call_global, ITERATIONS));
	printf("%f\n", run_async(totalThreads, numBlocks, blockSize, coordinates, BUFFER_SIZE, kernel_call_global, ITERATIONS));

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
__host__ void setup(
	const int totalThreads,
	const int numBlocks,
	const int blockSize,
	float *buffer)
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
/// @brief Executes a CUDA kernel in a serial memory access.
/// @param totalThreads Number of CUDA Threads
/// @param numBlocks Number of CUDA Blocks
/// @param blockSize CUDA Block Size
/// @param buffer Pointer to host buffer
/// @param buffer_size Size of the host buffer in bytes
/// @param kernel_function Pointer to the KERNEL_FUNCTION to execute.
/// @param interations Number of times to iterate the kernel_function.
/// @return The duration of time to execute the kernel function
//-----------------------------------------------------------------------------
__host__ float run_serial(
	const int totalThreads,
	const int numBlocks,
	const int blockSize,
	float *buffer,
	const size_t buffer_size,
	KERNEL_FUNCTION kernel_function,
	size_t iterations)
{
	// Setup stopwatch
	cudaEvent_t start;
	cudaEvent_t end;
	float timer;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Allocate host and device buffer
	float *host_buffer = (float *)malloc(buffer_size);
	float *dev_buffer;
	cudaMalloc(&dev_buffer, buffer_size);

	// Execute memory operations and the kernel function
	cudaEventRecord(start);
	while (iterations-- > 0)
	{
		cudaMemcpy(dev_buffer, buffer, buffer_size, cudaMemcpyHostToDevice);
		kernel_function<<<numBlocks, blockSize>>>(dev_buffer, totalThreads);
		cudaDeviceSynchronize();
		cudaMemcpy(host_buffer, dev_buffer, buffer_size, cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(end);

#if PRINT_RESULTS
	for (size_t idx = 0; idx < totalThreads; idx++)
		printf("(%f,%f)\n", host_buffer[idx], host_buffer[idx + totalThreads]);
#endif

	// Cleanup and record time
	free(host_buffer);
	cudaFree(dev_buffer);
	cudaEventElapsedTime(&timer, start, end);
	return timer;
}

//-----------------------------------------------------------------------------
/// @brief Executes a CUDA kernel in using async (stream) memory access.
/// @param totalThreads Number of CUDA Threads
/// @param numBlocks Number of CUDA Blocks
/// @param blockSize CUDA Block Size
/// @param buffer Pointer to host buffer
/// @param buffer_size Size of the host buffer in bytes
/// @param kernel_function Pointer to the KERNEL_FUNCTION to execute.
/// @param interations Number of times to iterate the kernel_function.
/// @return The duration of time to execute the kernel function
//-----------------------------------------------------------------------------
__host__ float run_async(
	const int totalThreads,
	const int numBlocks,
	const int blockSize,
	float *buffer,
	const size_t buffer_size,
	KERNEL_FUNCTION kernel_function,
	size_t iterations)
{
	// Setup stopwatch
	cudaEvent_t start;
	cudaEvent_t end;
	float timer;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Allocate host and device buffers
	float *host_input;
	float *host_result;
	float *dev_buffer;
	cudaHostAlloc((void **)&host_input, buffer_size, cudaHostAllocDefault);
	cudaHostAlloc((void **)&host_result, buffer_size, cudaHostAllocDefault);
	cudaMalloc(&dev_buffer, buffer_size);

	// The vanilla data has to copied to page-locked host memory
	memcpy(host_input, buffer, buffer_size);

	// Create Stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Execute memory operations and the kernel function
	cudaEventRecord(start);
	while (iterations-- > 0)
	{
		cudaMemcpyAsync(dev_buffer, host_input, buffer_size, cudaMemcpyHostToDevice, stream);
		kernel_function<<<numBlocks, blockSize, 1, stream>>>(dev_buffer, totalThreads);
		cudaMemcpyAsync(host_result, dev_buffer, buffer_size, cudaMemcpyDeviceToHost, stream);
	}
	cudaStreamSynchronize(stream);
	cudaEventRecord(end);

#if PRINT_RESULTS
	for (size_t idx = 0; idx < totalThreads; idx++)
		printf("(%f,%f)\n", host_result[idx], host_result[idx + totalThreads]);
#endif

	// Cleanup and record time
	cudaFreeHost(host_input);
	cudaFreeHost(host_result);
	cudaFree(dev_buffer);
	cudaEventElapsedTime(&timer, start, end);
	return timer;
}
