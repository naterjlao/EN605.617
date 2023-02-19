//-----------------------------------------------------------------------------
/// @file fileio.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 4 File Input/Output Implementation
//-----------------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include "fileio.h"

//-----------------------------------------------------------------------------
/// @brief
/// @param filename
/// @param buffer
/// @return
//-----------------------------------------------------------------------------
size_t read_file(const char *filename, char **buffer)
{
    size_t length = 0;
    FILE *file = fopen(filename, "r");

    if (file > 0)
    {
        // Get the length of the file
        fseek(file, 0, SEEK_END);
        length = ftell(file);
        fseek(file, 0, SEEK_SET);

        // blah
#ifndef __NVCC__
        *buffer = (char *)malloc(length);
#else
        cudaMallocHost(buffer, length);
#endif
        fread(*buffer, sizeof(char), length, file);
        fclose(file);
    }
    else
    {
        // File invalid, set buffer and length to null
        length = 0;
        *buffer = 0;
    }

    return length;
}

//-----------------------------------------------------------------------------
/// @brief
/// @param buffer
//-----------------------------------------------------------------------------
void free_buffer(char **buffer)
{
    if (*buffer != 0)
    {
#ifndef __NVCC__
        free(*buffer);
#else
        cudaFree(*buffer);
#endif
        *buffer = 0;
    }
}

//-----------------------------------------------------------------------------
/// @brief
/// @param filename
/// @param buffer
/// @param length
/// @return
//-----------------------------------------------------------------------------
bool write_file(const char *filename, char *buffer, const size_t length)
{
    bool retval = false;
    FILE *file = fopen(filename, "w");

    if (file != 0)
    {
        retval = fwrite(buffer, sizeof(char), length, file);
        fclose(file);
    }

    return retval;
}