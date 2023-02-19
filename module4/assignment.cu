#include <stdio.h>
#include "fileio.h"


int main(int argc, char** argv)
{
	// Evaluate arguments
	if (argc != 2)
	{
		printf("ERROR: invalid arguments, expecting:\n");
		printf("main <FILENAME>\n");
		return -1;
	}

	// Read file input
	const char *filename = argv[1];
	char *buffer;
	size_t length = read_file(filename, &buffer);
	if (length == 0)
	{
		printf("ERROR: empty file or file open failed");
		return -1;
	}

	// Setup Host and Device Memory
	const int BLOCK_SIZE = 256;
	const int NUM_BLOCKS = length / BLOCK_SIZE + ((length % BLOCK_SIZE > 0) ? 1 : 0);
	printf("Processing %s\n",filename);
	printf("Character Length=%d\n",length);
	printf("Allocating %d Blocks of Size %d\n",NUM_BLOCKS, BLOCK_SIZE);

	char *device_buffer;
	cudaMalloc(&device_buffer, (NUM_BLOCKS * BLOCK_SIZE) * sizeof(char)); // We allocate more if needed
	cudaMemcpy(device_buffer, buffer, length, cudaMemcpyHostToDevice);
#if 0
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = max(maxError, abs(y[i]-4.0f));
    printf("y[%d]=%f\n",i,y[i]);
  }
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);

#endif

	// Cleanup
	if (buffer != 0)
	{
		free(buffer);
	}

	return 0;
}
