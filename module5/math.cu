//-----------------------------------------------------------------------------
/// @file math.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 3 Math Functions
//-----------------------------------------------------------------------------
#include "math.h"

//-----------------------------------------------------------------------------
/// @brief Performs a vector addition on input arrays a and b.
/// @param a Input array vector.
/// @param b Input array vector.
/// @param out Output array containing the vector sum.
/// @param n Length of the array.
/// @return None; out is modified.
/// @note GPU Kernel Operation.
//-----------------------------------------------------------------------------
void list_add(const int *a, const int *b, int *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

//-----------------------------------------------------------------------------
/// @brief Performs a vector subtraction on input arrays a and b.
/// @param a Input array vector.
/// @param b Input array vector.
/// @param out Output array containing the vector difference.
/// @param n Length of the array.
/// @return None; out is modified.
/// @note GPU Kernel Operation.
//-----------------------------------------------------------------------------
void list_sub(const int *a, const int *b, int *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] - b[i];
}

//-----------------------------------------------------------------------------
/// @brief Performs a vector multiplication on input arrays a and b.
/// @param a Input array vector.
/// @param b Input array vector.
/// @param out Output array containing the vector product.
/// @param n Length of the array.
/// @return None; out is modified.
/// @note GPU Kernel Operation.
//-----------------------------------------------------------------------------
void list_mul(const int *a, const int *b, int *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] * b[i];
}

//-----------------------------------------------------------------------------
/// @brief Performs a vector modulo on input arrays a and b.
/// @param a Input array vector.
/// @param b Input array vector.
/// @param out Output array containing the result of element wise, a mod b.
/// @param n Length of the array.
/// @return None; out is modified.
/// @note CPU Host Operation.
//-----------------------------------------------------------------------------
void list_mod(const int *a, const int *b, int *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (b[i] > 0) ? (a[i] % b[i]) : 0;
}

//-----------------------------------------------------------------------------
/// @brief Performs a vector addition on input arrays a and b.
/// @param a Input array vector.
/// @param b Input array vector.
/// @param out Output array containing the vector sum.
/// @return None; out is modified.
/// @note GPU Kernel Operation.
//-----------------------------------------------------------------------------
__global__ void cuda_add(const int *a, const int *b, int *out)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = a[thread_idx] + b[thread_idx];
}

//-----------------------------------------------------------------------------
/// @brief Performs a vector subtraction on input arrays a and b.
/// @param a Input array vector.
/// @param b Input array vector.
/// @param out Output array containing the vector difference.
/// @return None; out is modified.
/// @note GPU Kernel Operation.
//-----------------------------------------------------------------------------
__global__ void cuda_sub(const int *a, const int *b, int *out)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = a[thread_idx] - b[thread_idx];
}

//-----------------------------------------------------------------------------
/// @brief Performs a vector multiplication on input arrays a and b.
/// @param a Input array vector.
/// @param b Input array vector.
/// @param out Output array containing the vector product.
/// @return None; out is modified.
/// @note GPU Kernel Operation.
//-----------------------------------------------------------------------------
__global__ void cuda_mul(const int *a, const int *b, int *out)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = a[thread_idx] * b[thread_idx];
}

//-----------------------------------------------------------------------------
/// @brief Performs a vector modulo on input arrays a and b.
/// @param a Input array vector.
/// @param b Input array vector.
/// @param out Output array containing the result of element wise, a mod b.
/// @return None; out is modified.
/// @note GPU Kernel Operation.
//-----------------------------------------------------------------------------
__global__ void cuda_mod(const int *a, const int *b, int *out)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = (b[thread_idx] > 0) ? (a[thread_idx] % b[thread_idx]) : 0;
}