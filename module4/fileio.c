#include <stdlib.h>
#include <stdio.h>
#include "fileio.h"

size_t read_file(const char *filename, char **buffer)
{
    size_t length = 0;
    FILE *file;
    char ch;

    file = fopen(filename, "r");

    if (file > 0)
    {
        // Get the length of the file
        fseek(file, 0, SEEK_END);
        length = ftell(file);
        fseek(file, 0, SEEK_SET);

        // Allocate memory
        *buffer = malloc(length);
        fread(*buffer, length, 1, file);
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