#include "transform.h"

__device__ void translate_2d(float *x, float *y, const float dx, const float dy)
{
    *x += dx;
    *y += dy;
}

__device__ void rotate_2d(float *x, float *y, const float rad)
{
    const float rx = (*x) * (cos(rad)) - (*y) * (sin(rad));
    const float ry = (*x) * (sin(rad)) + (*y) * (cos(rad));
    *x = rx;
    *y = ry;
}

__device__ void scale_rd(float *x, float *y, const float factor)
{
    *x *= factor;
    *y *= factor;
}