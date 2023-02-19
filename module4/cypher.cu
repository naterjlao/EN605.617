#include "cypher.h"

__global__
void caesar_cypher(char *buffer, const int offset)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    buffer[thread_idx] = buffer[thread_idx] + offset;
}