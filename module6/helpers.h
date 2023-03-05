//-----------------------------------------------------------------------------
/// @file helpers.h
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 5 Helper Functions
//-----------------------------------------------------------------------------
#ifndef __HELPERS_H__
#define __HELPERS_H__

#include <curand.h>
#include <curand_kernel.h>

__global__ void populate_thread_idx(int *l);
__global__ void setup_random(curandState *rand_state);
__global__ void populate_random_floats(float *l, curandState *rand_state);

#endif