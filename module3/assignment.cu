
#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
#include "math.h"

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	// Determine size of input
	const size_t ARRAY_LENGTH = totalThreads;
	const size_t ARRAY_BYTES = sizeof(int) * ARRAY_LENGTH;
	printf("Performing Operations on %d elements\n", ARRAY_LENGTH);
	printf("Number of Blocks = %d\n", numBlocks);
	printf("Block Size = %d\n", blockSize);

	// Allocate Arrays
	int *a = (int *) malloc(ARRAY_BYTES);
	int *b = (int *) malloc(ARRAY_BYTES);
	int *sum = (int *) malloc(ARRAY_BYTES);
	int *dif = (int *) malloc(ARRAY_BYTES);
	int *prd = (int *) malloc(ARRAY_BYTES);
	int *rem = (int *) malloc(ARRAY_BYTES);

	// Clear out arrays
	memset(a, 0, ARRAY_BYTES);
	memset(b, 0, ARRAY_BYTES);
	memset(sum, 0, ARRAY_BYTES);
	memset(dif, 0, ARRAY_BYTES);
	memset(prd, 0, ARRAY_BYTES);
	memset(rem, 0, ARRAY_BYTES);
	
	printf("Populating Random Array\n");
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
	cudaMemcpy( gpu_a, a, ARRAY_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_b, b, ARRAY_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_sum, sum, ARRAY_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_dif, dif, ARRAY_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_prd, prd, ARRAY_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_rem, rem, ARRAY_BYTES, cudaMemcpyHostToDevice );

	// GPU KERNEL EXECUTION
	printf("Populating Thread Idx Array\n");
	populate_thread_idx<<<numBlocks, blockSize>>>(gpu_a);

	printf("Performing GPU Addition\n");
	cuda_add<<<numBlocks, blockSize>>>(gpu_a, gpu_b, gpu_sum);

	printf("Performing GPU Subtraction\n");
	cuda_sub<<<numBlocks, blockSize>>>(gpu_a, gpu_b, gpu_dif);

	printf("Performing GPU Multiplication\n");
	cuda_mul<<<numBlocks, blockSize>>>(gpu_a, gpu_b, gpu_prd);

	printf("Performing GPU Modulo\n");
	cuda_mod<<<numBlocks, blockSize>>>(gpu_a, gpu_b, gpu_rem);

	// Free memory on GPU, copy to Host
	cudaMemcpy( a, gpu_a, ARRAY_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( b, gpu_b, ARRAY_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( sum, gpu_sum, ARRAY_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( dif, gpu_dif, ARRAY_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( prd, gpu_prd, ARRAY_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( rem, gpu_rem, ARRAY_BYTES, cudaMemcpyDeviceToHost );
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_sum);
	cudaFree(gpu_dif);
	cudaFree(gpu_prd);
	cudaFree(gpu_rem);

	// CPU HOST EXECUTION
	int *cpu_sum = (int *) malloc(ARRAY_BYTES);
	int *cpu_dif = (int *) malloc(ARRAY_BYTES);
	int *cpu_prd = (int *) malloc(ARRAY_BYTES);
	int *cpu_rem = (int *) malloc(ARRAY_BYTES);

	printf("Performing CPU Addition\n");
	list_add(a, b, cpu_sum, ARRAY_LENGTH);

	printf("Performing CPU Subtraction\n");
	list_sub(a, b, cpu_sum, ARRAY_LENGTH);

	printf("Performing CPU Multiplication\n");
	list_mul(a, b, cpu_sum, ARRAY_LENGTH);

	printf("Performing CPU Modulo\n");
	list_mod(a, b, cpu_sum, ARRAY_LENGTH);

	for (size_t i = 0; i < ARRAY_LENGTH; i++)
	{
		//printf("%d %d %d %d %d\n",a[i], b[i], sum[i], dif[i], rem[i]);
	}

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
