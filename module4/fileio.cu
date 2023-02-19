//-----------------------------------------------------------------------------
/// @file fileio.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 4 File Input/Output Implementation
//-----------------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include "fileio.h"

//-----------------------------------------------------------------------------
/// @brief Reads in file contents to memory.
/// @param filename Name of the file to be read.
/// @param buffer Address of the buffer pointer.
/// @return The size in bytes of the content read.
/// @note If compiled in g++, Host Pageable Memory (malloc) is used. If compiled in
/// nvcc, CUDA Pinned Memory (cudaMallocHost) is used.
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
/// @brief Frees the allocated memory from the file buffer.
/// @param buffer Address to the buffer pointer.
//-----------------------------------------------------------------------------
void free_buffer(char **buffer)
{
    // Check if the buffer was allocated
    if (*buffer != 0)
    {
        // Call the appropriate free function
#ifndef __NVCC__
        free(*buffer);
#else
        cudaFree(*buffer);
#endif
        // Set the pointer to null
        *buffer = 0;
    }
}

//-----------------------------------------------------------------------------
/// @brief Writes a buffer to a file.
/// @param filename Name of the file to write.
/// @param buffer Buffer pointer.
/// @param length Length of the contents in the buffer.
/// @return True if the operation was successful; false otherwise.
/// @warning This function will overwrite existing files of the same name.
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