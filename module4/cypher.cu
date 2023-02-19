//-----------------------------------------------------------------------------
/// @file cypher.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 4 Caesar Cypher Implementation
//-----------------------------------------------------------------------------
#include "cypher.h"

//-----------------------------------------------------------------------------
/// @brief Offsets the character buffer by a given value.
/// @param buffer Pointer to character buffer.
/// @param offset Offset value.
/// @return None.
//-----------------------------------------------------------------------------
__global__ void caesar_cypher(char *buffer, const int offset)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    char ch = buffer[thread_idx];
    
    if (ch >= 'A' && ch <= 'Z')
    {
        ch += offset;
        ch += (ch < 'A') ? ('Z' - 'A' + 1) : 0;
        ch -= (ch > 'Z') ? ('Z' - 'A' + 1) : 0;
    }

    if (ch >= 'a' && ch <= 'z')
    {
        ch += offset;
        ch += (ch < 'a') ? ('z' - 'a' + 1) : 0;
        ch -= (ch > 'z') ? ('z' - 'a' + 1) : 0;
    }

    buffer[thread_idx] = ch;
}