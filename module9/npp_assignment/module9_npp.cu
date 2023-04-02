//-----------------------------------------------------------------------------
/// @file module9_npp.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 9 Main Driver
//-----------------------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <npp.h>

#define PRINT_DEBUG 0

//-----------------------------------------------------------------------------
/// @brief Main Driver for NPP example
/// @param argc
/// @param argv
/// @return 0
/// @note This is based off from examples from:
/// - https://docs.nvidia.com/cuda/npp/general_conventions_lb.html
/// - https://stackoverflow.com/questions/68873415/cuda-npp-min-max-returns-wrong-output
//-----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // Read command line arguments
    size_t dim_size = (1 << 20);
    int iterations = 1000;

    if (argc >= 2)
    {
        dim_size = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        iterations = atoi(argv[2]);
    }

    // Get buffer size for the scratch-memory buffer
    int BufferSize = 0;
    nppsMinMaxGetBufferSize_32f(dim_size, &BufferSize);

    // Allocate memory on the GPU
    Npp32f *d_img;
    Npp32f *dMax, *dMin;
    Npp8u *pScratch;
    cudaMalloc((void **)&d_img, dim_size * sizeof(Npp32f));
    cudaMalloc((void **)&dMax, sizeof(Npp32f));
    cudaMalloc((void **)&dMin, sizeof(Npp32f));
    cudaMalloc((void **)(&pScratch), BufferSize);

    // Fill out with ones
    nppsSet_32f(1.0f, d_img, dim_size);

    // Perform Min-Max
    std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++)
        nppsMinMax_32f(d_img, dim_size, dMin, dMax, pScratch);
    std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();

#if PRINT_DEBUG
    Npp32f Max, Min;
    cudaMemcpy(&Max, dMax, sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Min, dMin, sizeof(Npp32f), cudaMemcpyDeviceToHost);
    std::cout << "BufferSize: " << BufferSize << " ";
    std::cout << "Max: " << Max << " ";
    std::cout << "Min: " << Min << std::endl;
#endif

    // Print performance metrics
    std::cout << dim_size << ", " << iterations << ", ";
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << std::endl;

    cudaFree(d_img);
    cudaFree(pScratch);
    cudaFree(dMax);
    cudaFree(dMin);

    return 0;
}