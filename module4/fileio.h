//-----------------------------------------------------------------------------
/// @file fileio.h
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 4 File Input/Output Headers
//-----------------------------------------------------------------------------
#ifndef __FILEIO_H__
#define __FILEIO_H__

#include <stdlib.h>

size_t read_file(const char *filename, char **buffer);
void free_buffer(char **buffer);
bool write_file(const char *filename, char *buffer, const size_t length);

#endif