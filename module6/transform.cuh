#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__
#include <math.h>

__device__ void translate_2d(float *x, float *y, const float dx, const float dy);
__device__ void rotate_2d(float *x, float *y, const float rad);
__device__ void scale_rd(float *x, float *y, const float factor);

#endif