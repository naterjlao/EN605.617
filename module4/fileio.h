#ifndef __FILEIO_H__
#define __FILEIO_H__

#include <stdlib.h>

extern "C" size_t read_file(const char *filename, char **buffer);

#endif