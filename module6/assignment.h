#ifndef __ASSIGNMENT_H__
#define __ASSIGNMENT_H__

#define NUM_ELEMENTS 4
__global__ void kernel_call_const(float *buffer);
__global__ void kernel_call_shared(float *buffer);
__host__ void setup(const int totalThreads, const int numBlocks, const int blockSize, float * buffer);
__device__ float operation(float val, float arr_addr[NUM_ELEMENTS], float arr_subb[NUM_ELEMENTS], float arr_mult[NUM_ELEMENTS], float arr_divd[NUM_ELEMENTS]);
__device__ float math(const float i, const float add, const float sub, const float mul, const float div);

#endif