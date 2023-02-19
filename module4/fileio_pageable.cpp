#include <stdlib.h>
#include <stdio.h>
#include "fileio.h"

extern "C" size_t read_file(const char *filename, char **buffer)
{
    size_t length = 0;
    FILE *file = fopen(filename, "r");

    if (file > 0)
    {
        // Get the length of the file
        fseek(file, 0, SEEK_END);
        length = ftell(file);
        fseek(file, 0, SEEK_SET);

        // Allocate memory
        *buffer = (char *) malloc(length);
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