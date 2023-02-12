#include <stdlib.h>
#include "helpers.h"

/// @brief 
/// @param l 
/// @return 
/// @note GPU Kernel Operation
__global__ void populate_thread_idx(int *l)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    l[thread_idx] = thread_idx;
}

/// @brief 
/// @param l 
/// @param n 
/// @param max
/// @note CPU Host Operation 
void populate_random_list(int *l, size_t n, int max)
{
    /// @todo seed this
    /// @todo probably want to use cuda rand
    for (size_t i = 0; i < n; i++)
        l[i] = rand() % (max + 1);
}
