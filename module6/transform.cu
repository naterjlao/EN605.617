#include "transform.cuh"

// Kernel for const memory operations
__global__ void kernel_call_register(float *buffer, const size_t n_points)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float x = buffer[idx];
	float y = buffer[idx + n_points];
	operation(&x, &y);
	buffer[idx] = x;
	buffer[idx + n_points] = y;
}

// Kernel for shared memory operations
__global__ void kernel_call_global(float *buffer, const size_t n_points)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	operation(&buffer[idx], &buffer[idx + n_points]);
}

#define INTERATIONS 1024
__device__ void operation(float *x, float *y)
{
	for (int i = 0; i < INTERATIONS; i++)
	{
		normalize_2d(x, y);
		rotate_2d(x, y, M_PI * 0.15);
		translate_2d(x, y, 4.0, 2.68);
		scale_2d(x, y, 23.0);
	}
}

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

__device__ void scale_2d(float *x, float *y, const float factor)
{
    *x *= factor;
    *y *= factor;
}

__device__ void normalize_2d(float *x, float *y)
{
    float mag = sqrt(((*x) * (*x)) + ((*y) * (*y)));
    if (mag > 0.0 || mag < 0.0)
    {
        *x /= mag;
        *y /= mag;
    }
}