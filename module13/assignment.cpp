//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context0;
    cl_program program0;
    int * inputOutput0;

    int platform = DEFAULT_PLATFORM; 

    std::cout << "Simple buffer and sub-buffer Example" << std::endl;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("simple.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context0 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");
 
    // Create program from source
    program0 = clCreateProgramWithSource(
        context0, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
 
    // Build program
    errNum = clBuildProgram(
        program0,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
 
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog0[16384];
        clGetProgramBuildInfo(
            program0, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog0), 
            buildLog0, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog0;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    inputOutput0 = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput0[i] = i;
    }

    // create a single buffer to cover all the input data
    cl_mem buffer0 = clCreateBuffer(
        context0,
        CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // Create command queues
    InfoDevice<cl_device_type>::display(
     	deviceIDs[0], 
     	CL_DEVICE_TYPE, 
     	"CL_DEVICE_TYPE");

    cl_command_queue queue0 = 
     	clCreateCommandQueue(
     	context0,
     	deviceIDs[0],
     	0,
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");
 
    cl_kernel kernel0 = clCreateKernel(
     program0,
     "square",
     &errNum);
    checkErr(errNum, "clCreateKernel(square)");

    errNum = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&buffer0);
    checkErr(errNum, "clSetKernelArg(square)");
 
    // call kernel for each device
    cl_event event0;
    size_t gWI = NUM_BUFFER_ELEMENTS;
	
	// Create a user event
 	cl_event eventUser = clCreateUserEvent(context0, &errNum);
	checkErr(errNum, "clCreateUserEvent");

 	//Wait for queue 1 to complete before continuing on queue 0
 	errNum = clEnqueueBarrier(queue0);
 	errNum = clEnqueueWaitForEvents(queue0, 1, &eventUser);

	size_t userIdx = 0;
	while (userIdx < NUM_BUFFER_ELEMENTS)
	{
		std::cout << "input[" << userIdx << "] = ";
		std::cin >> inputOutput0[userIdx];
		userIdx++;
	}
	errNum = clSetUserEventStatus (eventUser,CL_COMPLETE);
	std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();

    // Write input data
    errNum = clEnqueueWriteBuffer(
      queue0,
      buffer0,
      CL_TRUE,
      0,
      sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
      (void*)inputOutput0,
      0,
      0,
      &event0);

    errNum = clEnqueueNDRangeKernel(
      queue0, 
      kernel0, 
      1, 
      NULL,
      (const size_t*)&gWI, 
      (const size_t*)NULL, 
      0, 
      0, 
      &event0);

 	// Read back computed data
   	clEnqueueReadBuffer(
            queue0,
            buffer0,
            CL_TRUE,
            0,
            sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)inputOutput0,
            0,
            NULL,
            NULL);

	std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();

    // Display output in rows
    for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++)
    {
     std::cout << " " << inputOutput0[elems];
    }
    std::cout << std::endl;
 
    std::cout << "Program completed successfully" << std::endl;
	std::cout << "Kernel Execution Time: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()
		<< "us" << std::endl;

    return 0;
}
