#ifndef __MATH_H__
#define __MATH_H__

void list_add(const int *a, const int *b, int *out, size_t n);
void list_sub(const int *a, const int *b, int *out, size_t n);
void list_mul(const int *a, const int *b, int *out, size_t n);
void list_mod(const int *a, const int *b, int *out, size_t n);

__global__ void cuda_add(const int *a, const int *b, int *out);
__global__ void cuda_sub(const int *a, const int *b, int *out);
__global__ void cuda_mul(const int *a, const int *b, int *out);
__global__ void cuda_mod(const int *a, const int *b, int *out);

#endif