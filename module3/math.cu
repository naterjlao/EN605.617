#include "math.h"

void list_add(const int *a, const int *b, int *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

void list_sub(const int *a, const int *b, int *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] - b[i];
}

void list_mul(const int *a, const int *b, int *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] * b[i];
}

void list_mod(const int *a, const int *b, int *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (b[i] > 0) ? (a[i] % b[i]) : 0;
}


/// @brief
/// @param a
/// @param b
/// @param out
/// @return
/// @note GPU Kernel Operation
__global__ void cuda_add(const int *a, const int *b, int *out)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = a[thread_idx] + b[thread_idx];
}

/// @brief
/// @param a
/// @param b
/// @param out
/// @return
/// @note GPU Kernel Operation
__global__ void cuda_sub(const int *a, const int *b, int *out)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = a[thread_idx] - b[thread_idx];
}

/// @brief
/// @param a
/// @param b
/// @param out
/// @return
/// @note GPU Kernel Operation
__global__ void cuda_mul(const int *a, const int *b, int *out)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = a[thread_idx] * b[thread_idx];
}

/// @brief
/// @param a
/// @param b
/// @param out
/// @return
/// @note GPU Kernel Operation
__global__ void cuda_mod(const int *a, const int *b, int *out)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = (b[thread_idx] > 0) ? (a[thread_idx] % b[thread_idx]) : 0;
}