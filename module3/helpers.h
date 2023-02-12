//-----------------------------------------------------------------------------
/// @file
/// @author Nate Lao (nlao1@jh.edu)
/// @brief
//-----------------------------------------------------------------------------
#ifndef __HELPERS_H__
#define __HELPERS_H__

__global__ void populate_thread_idx(int *l);

void populate_random_list(int *l, size_t n, int max);

#endif