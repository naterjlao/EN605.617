#include <stdio.h>
#include <chrono>
#include "helpers.h"

__global__ void cuda_branch(int *a)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_idx % 2 > 0)
    {
        a[thread_idx] = 1;
    }
    else
    {
        a[thread_idx] = 0;
    }
}

void host_branch(int *a, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (i % 2 > 0)
        {
            a[i] = 1;
        }
        else
        {
            a[i] = 0;
        }
    }
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

	std::chrono::time_point<std::chrono::steady_clock> start_time;
	std::chrono::time_point<std::chrono::steady_clock> stop_time;

	// validate command line arguments
	if (totalThreads % blockSize != 0)
	{
		++numBlocks;
		totalThreads = numBlocks * blockSize;
	}

    int *cpu_buff = (int *) malloc(totalThreads * sizeof(int));
    memset(cpu_buff, 0, (totalThreads * sizeof(int)));
    int *gpu_buff;
    cudaMalloc((void **)&gpu_buff, (totalThreads * sizeof(int)));

    printf("%d,%d,%d",totalThreads,numBlocks,blockSize);

    start_time = std::chrono::steady_clock::now();
    cuda_branch<<<numBlocks, blockSize>>>(gpu_buff);
	stop_time = std::chrono::steady_clock::now();
    printf("%d,",(stop_time - start_time));

	start_time = std::chrono::steady_clock::now();
    host_branch(cpu_buff, totalThreads);
	stop_time = std::chrono::steady_clock::now();
    printf("%d",(stop_time - start_time));

    printf("\n");

    free(cpu_buff);
    cudaFree(gpu_buff);
}