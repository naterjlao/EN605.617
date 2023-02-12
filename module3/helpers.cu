#include <stdlib.h>
#include "helpers.h"

void populate_random_list(int *l, size_t n, int max)
{
    /// @todo seed random generator
    for (size_t i = 0; i < n; i++)
        l[i] = rand() % (max + 1);
}
void foo() {}