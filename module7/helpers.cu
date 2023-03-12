//-----------------------------------------------------------------------------
/// @file helpers.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 5 Helper Functions
//-----------------------------------------------------------------------------
#include "helpers.cuh"

//-----------------------------------------------------------------------------
/// @brief Assigns the index number to each corresponding element in a given array.
/// @param l Array vector to populate.
/// @return None; l is modified.
/// @note GPU Kernel Operation.
//-----------------------------------------------------------------------------
__global__ void populate_thread_idx(int *l)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    l[idx] = idx;
}

__global__ void setup_random(curandState *rand_state)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(420, idx, 0, (rand_state + idx));
}

//-----------------------------------------------------------------------------
/// @brief Populates the given array vector with a random float between 0 to 1.0 (exclusive).
/// @param l Array vector to populate.
/// @return None; l is modified.
/// @note GPU Kernel Operation
//-----------------------------------------------------------------------------
__global__ void populate_random_floats(float *l, curandState *rand_state)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    l[idx] = curand_uniform(rand_state + idx);
}
