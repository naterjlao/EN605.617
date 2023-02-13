//-----------------------------------------------------------------------------
/// @file assignment.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 3 Assignment Driver
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <chrono>
#include "helpers.h"
#include "math.h"

void log(const char *format, ...)
{
#ifndef CSV
	va_list argptr;
	va_start(argptr, format);
	vprintf(format, argptr);
	va_end(argptr);
#endif
}

void csv(const char *format, ...)
{
#ifdef CSV
	va_list argptr;
	va_start(argptr, format);
	vprintf(format, argptr);
	va_end(argptr);
#endif
}

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

	log("INITIALIZING\n");
	std::chrono::time_point<std::chrono::steady_clock> start_time;
	std::chrono::time_point<std::chrono::steady_clock> stop_time;

	// validate command line arguments
	if (totalThreads % blockSize != 0)
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;

		log("Warning: Total thread count is not evenly divisible by the block size\n");
		log("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	// Determine size of input
	const size_t ARRAY_LENGTH = totalThreads;
	const size_t ARRAY_BYTES = sizeof(int) * ARRAY_LENGTH;
	log("Performing Operations on %d elements\n", ARRAY_LENGTH);
	log("Number of Blocks = %d\n", numBlocks);
	log("Block Size = %d\n", blockSize);
	csv("%d,",ARRAY_LENGTH);
	csv("%d,",numBlocks);
	csv("%d,",blockSize);

	// Allocate Arrays
	int *a = (int *)malloc(ARRAY_BYTES);
	int *b = (int *)malloc(ARRAY_BYTES);
	int *sum = (int *)malloc(ARRAY_BYTES);
	int *dif = (int *)malloc(ARRAY_BYTES);
	int *prd = (int *)malloc(ARRAY_BYTES);
	int *rem = (int *)malloc(ARRAY_BYTES);

	// Clear out arrays
	memset(a, 0, ARRAY_BYTES);
	memset(b, 0, ARRAY_BYTES);
	memset(sum, 0, ARRAY_BYTES);
	memset(dif, 0, ARRAY_BYTES);
	memset(prd, 0, ARRAY_BYTES);
	memset(rem, 0, ARRAY_BYTES);

	log("Populating Random Array\n");
	populate_random_list(b, ARRAY_LENGTH, 3);

	// Allocate GPU Arrays
	int *gpu_a;
	int *gpu_b;
	int *gpu_sum;
	int *gpu_dif;
	int *gpu_prd;
	int *gpu_rem;

	// Allocate memory on GPU, copy from Host
	cudaMalloc((void **)&gpu_a, ARRAY_BYTES);
	cudaMalloc((void **)&gpu_b, ARRAY_BYTES);
	cudaMalloc((void **)&gpu_sum, ARRAY_BYTES);
	cudaMalloc((void **)&gpu_dif, ARRAY_BYTES);
	cudaMalloc((void **)&gpu_prd, ARRAY_BYTES);
	cudaMalloc((void **)&gpu_rem, ARRAY_BYTES);
	cudaMemcpy(gpu_a, a, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_sum, sum, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_dif, dif, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_prd, prd, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_rem, rem, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// GPU KERNEL EXECUTION
	log("Populating Thread Idx Array\n");
	populate_thread_idx<<<numBlocks, blockSize>>>(gpu_a);

	log("\nGPU EXECUTE\n");

	log("Performing GPU Addition");
	start_time = std::chrono::steady_clock::now();
	cuda_add<<<numBlocks, blockSize>>>(gpu_a, gpu_b, gpu_sum);
	stop_time = std::chrono::steady_clock::now();
	log(" ---------------------- %d Ticks\n", (stop_time - start_time));
	csv("%d,",(stop_time - start_time));

	log("Performing GPU Subtraction");
	cuda_sub<<<numBlocks, blockSize>>>(gpu_a, gpu_b, gpu_dif);
	stop_time = std::chrono::steady_clock::now();
	log(" ------------------- %d Ticks\n", (stop_time - start_time));
	csv("%d,",(stop_time - start_time));

	log("Performing GPU Multiplication");
	cuda_mul<<<numBlocks, blockSize>>>(gpu_a, gpu_b, gpu_prd);
	stop_time = std::chrono::steady_clock::now();
	log(" ---------------- %d Ticks\n", (stop_time - start_time));
	csv("%d,",(stop_time - start_time));

	log("Performing GPU Modulo");
	cuda_mod<<<numBlocks, blockSize>>>(gpu_a, gpu_b, gpu_rem);
	stop_time = std::chrono::steady_clock::now();
	log(" ------------------------ %d Ticks\n", (stop_time - start_time));
	csv("%d,",(stop_time - start_time));

	// Free memory on GPU, copy to Host
	cudaMemcpy(a, gpu_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, gpu_b, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(sum, gpu_sum, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(dif, gpu_dif, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(prd, gpu_prd, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(rem, gpu_rem, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_sum);
	cudaFree(gpu_dif);
	cudaFree(gpu_prd);
	cudaFree(gpu_rem);

	// CPU HOST EXECUTION
	int *cpu_sum = (int *)malloc(ARRAY_BYTES);
	int *cpu_dif = (int *)malloc(ARRAY_BYTES);
	int *cpu_prd = (int *)malloc(ARRAY_BYTES);
	int *cpu_rem = (int *)malloc(ARRAY_BYTES);

	log("\nCPU EXECUTE\n");

	log("Performing CPU Addition");
	start_time = std::chrono::steady_clock::now();
	list_add(a, b, cpu_sum, ARRAY_LENGTH);
	stop_time = std::chrono::steady_clock::now();
	log(" ---------------------- %d Ticks\n", (stop_time - start_time));
	csv("%d,",(stop_time - start_time));

	log("Performing CPU Subtraction");
	start_time = std::chrono::steady_clock::now();
	list_sub(a, b, cpu_dif, ARRAY_LENGTH);
	stop_time = std::chrono::steady_clock::now();
	log(" ------------------- %d Ticks\n", (stop_time - start_time));
	csv("%d,",(stop_time - start_time));

	log("Performing CPU Multiplication");
	start_time = std::chrono::steady_clock::now();
	list_mul(a, b, cpu_prd, ARRAY_LENGTH);
	stop_time = std::chrono::steady_clock::now();
	log(" ---------------- %d Ticks\n", (stop_time - start_time));
	csv("%d,",(stop_time - start_time));

	log("Performing CPU Modulo");
	start_time = std::chrono::steady_clock::now();
	list_mod(a, b, cpu_rem, ARRAY_LENGTH);
	stop_time = std::chrono::steady_clock::now();
	log(" ------------------------ %d Ticks\n", (stop_time - start_time));
	csv("%d\n",(stop_time - start_time));

	// Verify if results are accurate
	log("\nTEST RESULTS\n");
	bool add_test = true;
	bool sub_test = true;
	bool mul_test = true;
	bool mod_test = true;
	for (size_t i = 0; i < ARRAY_LENGTH; i++)
	{
		add_test = add_test && (sum[i] == cpu_sum[i]);
		sub_test = add_test && (dif[i] == cpu_dif[i]);
		mul_test = add_test && (prd[i] == cpu_prd[i]);
		mod_test = add_test && (rem[i] == cpu_rem[i]);
	}
	log("Addition ------------------------------------- %s\n", add_test ? "PASSED" : "FAILED");
	log("Substraction --------------------------------- %s\n", sub_test ? "PASSED" : "FAILED");
	log("Multiplication ------------------------------- %s\n", mul_test ? "PASSED" : "FAILED");
	log("Modulo --------------------------------------- %s\n", mod_test ? "PASSED" : "FAILED");

	free(a);
	free(b);
	free(sum);
	free(dif);
	free(prd);
	free(rem);
	free(cpu_sum);
	free(cpu_dif);
	free(cpu_prd);
	free(cpu_rem);
}
