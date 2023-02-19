//-----------------------------------------------------------------------------
/// @file cypher.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 4 Caesar Cypher Implementation
//-----------------------------------------------------------------------------
#include "cypher.h"

//-----------------------------------------------------------------------------
/// @brief 
/// @param buffer 
/// @param offset 
/// @return 
//-----------------------------------------------------------------------------
__global__ void caesar_cypher(char *buffer, const int offset)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    buffer[thread_idx] = buffer[thread_idx] + offset;
}