//-----------------------------------------------------------------------------
/// @file transform.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 6 Matrix Transformation Functions
//-----------------------------------------------------------------------------
#include "transform.cuh"

//-----------------------------------------------------------------------------
/// @brief Kernel call for Global Memory access
/// @param buffer Pointer to host buffer
/// @param n_points Number of points in in coordinate buffer. Note that the buffer
/// itself must have 2*n_points float elements.
/// @return None
//-----------------------------------------------------------------------------
__global__ void kernel_call_global(float *buffer, const size_t n_points)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	operation(&buffer[idx], &buffer[idx + n_points]);
}

//-----------------------------------------------------------------------------
/// @brief Kernel call for Register Memory access
/// @param buffer Pointer to host buffer
/// @param n_points Number of points in in coordinate buffer. Note that the buffer
/// itself must have 2*n_points float elements.
/// @return None
//-----------------------------------------------------------------------------
__global__ void kernel_call_register(float *buffer, const size_t n_points)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float x = buffer[idx];
	float y = buffer[idx + n_points];
	operation(&x, &y);
	buffer[idx] = x;
	buffer[idx + n_points] = y;
}

/// @brief
/// @param x
/// @param y
/// @return
__device__ void operation(float *x, float *y)
{
	// Normalize is bogus, scale twice
	rotate_2d(x, y, M_PI * 0.15);
	translate_2d(x, y, 4.0, 2.68);
	scale_2d(x, y, 23.0);
	scale_2d(x, y, 1.0 / 23.0);
}

/// @brief
/// @param x
/// @param y
/// @param dx
/// @param dy
/// @return None
__device__ void translate_2d(float *x, float *y, const float dx, const float dy)
{
	*x += dx;
	*y += dy;
}

/// @brief
/// @param x
/// @param y
/// @param rad
/// @return None
__device__ void rotate_2d(float *x, float *y, const float rad)
{
	const float rx = (*x) * (cos(rad)) - (*y) * (sin(rad));
	const float ry = (*x) * (sin(rad)) + (*y) * (cos(rad));
	*x = rx;
	*y = ry;
}

/// @brief
/// @param x
/// @param y
/// @param factor
/// @return None
__device__ void scale_2d(float *x, float *y, const float factor)
{
	*x *= factor;
	*y *= factor;
}

/// @brief
/// @param x
/// @param y
/// @return None
__device__ void normalize_2d(float *x, float *y)
{
	float mag = sqrt(((*x) * (*x)) + ((*y) * (*y)));
	if (mag > 0.0 || mag < 0.0)
	{
		*x /= mag;
		*y /= mag;
	}
}