//-----------------------------------------------------------------------------
/// @file transform.cuh
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 7 Matrix Transformation Functions
//-----------------------------------------------------------------------------
#ifndef __TRANSFORM_CUH__
#define __TRANSFORM_CUH__
#include <math.h>

__global__ void kernel_call_global(float *buffer, const size_t n_points);
__global__ void kernel_call_register(float *buffer, const size_t n_points);
__device__ void operation(float *x, float *y);
__device__ void translate_2d(float *x, float *y, const float dx, const float dy);
__device__ void rotate_2d(float *x, float *y, const float rad);
__device__ void scale_2d(float *x, float *y, const float factor);
__device__ void normalize_2d(float *x, float *y);

#endif