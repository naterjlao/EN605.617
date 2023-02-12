//-----------------------------------------------------------------------------
/// @file helpers.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 3 Helper Functions
//-----------------------------------------------------------------------------
#include <stdlib.h>
#include "helpers.h"

//-----------------------------------------------------------------------------
/// @brief Assigns the index number to each corresponding element in a given array.
/// @param l Array vector to populate.
/// @return None; l is modified.
/// @note GPU Kernel Operation.
//-----------------------------------------------------------------------------
__global__ void populate_thread_idx(int *l)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    l[thread_idx] = thread_idx;
}

//-----------------------------------------------------------------------------
/// @brief Populates the given array vector with a random number between 0 to max.
/// @param l Array vector to populate.
/// @param n Length of the array vector.
/// @param max Max limit of the random range.
/// @note CPU Host Operation.
//-----------------------------------------------------------------------------
void populate_random_list(int *l, size_t n, int max)
{
    /// @todo seed this
    /// @todo probably want to use cuda rand
    for (size_t i = 0; i < n; i++)
        l[i] = rand() % (max + 1);
}
