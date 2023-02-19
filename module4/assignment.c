#include <stdio.h>
#include "fileio.h"

int main(int argc, char** argv)
{
	// Evaluate arguments
	if (argc != 2)
	{
		printf("ERROR: invalid arguments, expecting:\n");
		printf("main <FILENAME>\n");
		return -1;
	}

	// Read file input
	const char *filename = argv[1];
	char *buffer;
	size_t length = read_file(filename, &buffer);
	if (length == 0)
	{
		printf("ERROR: empty file or file open failed");
		return -1;
	}


	// Cleanup
	if (buffer != 0)
	{
		free(buffer);
	}

	return 0;
}
