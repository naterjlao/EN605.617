//-----------------------------------------------------------------------------
/// @file assignment.cpp
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 10 Main Driver
/// @note This is derived from the original HelloWorld.cpp example provided by:
///
/// Book:      OpenCL(R) Programming Guide
/// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
/// ISBN-10:   0-321-74964-2
/// ISBN-13:   978-0-321-74964-2
/// Publisher: Addison-Wesley Professional
/// URLs:      http://safari.informit.com/9780132488006/
///            http://www.openclprogrammingguide.com
///
//-----------------------------------------------------------------------------
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define PRINT_DEBUG 0

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)firstPlatformId,
            0};
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete[] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete[] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b, const size_t arraySize)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * arraySize, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * arraySize, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * arraySize, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);
}

int64_t ExecuteKernel(
    cl_context context,
    cl_command_queue commandQueue,
    cl_program program,
    cl_mem memObjects[3],
    const char *kernelFunction,
    const size_t arraySize,
    float *result)
{
    int64_t executionTime = -1;
    cl_kernel kernel = 0;
    cl_int errNum = CL_SUCCESS;

    // Create OpenCL kernel
    kernel = clCreateKernel(program, kernelFunction, NULL);
    if (kernel == NULL)
    {
        errNum = CL_BUILD_PROGRAM_FAILURE;
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
    }

    if (errNum == CL_SUCCESS)
    {
        // Set the kernel arguments (result, a, b)
        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error setting kernel arguments." << std::endl;
            Cleanup(context, commandQueue, program, kernel, memObjects);
        }
    }

    // Measure Execution Time from before the kernel queue to after the result has been returned
    // to the Host.
    std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
    if (errNum == CL_SUCCESS)
    {
        size_t globalWorkSize[1] = {arraySize};
        size_t localWorkSize[1] = {1};

        // Queue the kernel up for execution across the array
        errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                        globalWorkSize, localWorkSize,
                                        0, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error queuing kernel for execution." << std::endl;
            Cleanup(context, commandQueue, program, kernel, memObjects);
        }
    }

    if (errNum == CL_SUCCESS)
    {
        // Read the output buffer back to the Host
        errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                    0, arraySize * sizeof(float), result,
                                    0, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error reading result buffer." << std::endl;
            Cleanup(context, commandQueue, program, kernel, memObjects);
        }
    }
    std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();

    if (errNum == CL_SUCCESS)
    {
        executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    }

    return executionTime;
}

/// @brief Main Driver
/// @param argc 
/// @param argv 
/// @return 
int main(int argc, char **argv)
{
    // Optional arugment: arraySize
    size_t arraySize = 2 << 10;
    if (argc >= 2)
    {
        arraySize = atoi(argv[1]);
    }

    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_mem memObjects[3] = {0, 0, 0};
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, 0, memObjects);
        return 1;
    }

    // Create OpenCL program from kernel source
    program = CreateProgram(context, device, "math.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue, program, 0, memObjects);
        return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float result[arraySize];
    float a[arraySize];
    float b[arraySize];
    memset(result, 0, sizeof(float) * arraySize);
    for (int i = 0; i < arraySize; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    if (!CreateMemObjects(context, memObjects, a, b, arraySize))
    {
        Cleanup(context, commandQueue, program, 0, memObjects);
        return 1;
    }

    // Perform Kernel Executions
    std::cout << arraySize << ",";
    std::cout << ExecuteKernel(context, commandQueue, program, memObjects, "add_kernel", arraySize, result) << ",";
    std::cout << ExecuteKernel(context, commandQueue, program, memObjects, "sub_kernel", arraySize, result) << ",";
    std::cout << ExecuteKernel(context, commandQueue, program, memObjects, "mul_kernel", arraySize, result) << ",";
    std::cout << ExecuteKernel(context, commandQueue, program, memObjects, "pow_kernel", arraySize, result) << std::endl;

#if PRINT_DEBUG
    // Output the result buffer
    for (size_t i = 0; i < arraySize; i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
#endif
    Cleanup(context, commandQueue, program, 0, memObjects);

    return 0;
}
