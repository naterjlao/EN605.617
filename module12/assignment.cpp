
//-----------------------------------------------------------------------------
/// @file assignment.cpp
/// @author Nate Lao (nlao1@jh.edu)
/// @brief Module 12 Main Driver
/// @note This is derived from example provided by:
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
checkErr(cl_int err, const char *name)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

// Main Driver
int main(int argc, char **argv)
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id *platformIDs;
	cl_device_id *deviceIDs;
	cl_context context;
	cl_program program;
	std::vector<cl_kernel> kernels;
	std::vector<cl_command_queue> queues;
	std::vector<cl_mem> buffers;
	int *inputOutput;
	size_t iterations = 1;
	if (argc >= 2)
	{
		iterations = atoi(argv[1]);
	}

	int platform = DEFAULT_PLATFORM;

	// First, select an OpenCL platform to run on.
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	platformIDs = (cl_platform_id *)alloca(
		sizeof(cl_platform_id) * numPlatforms);

#if PRINT_DEBUG
	std::cout << "Number of platforms: \t" << numPlatforms << std::endl;
#endif

	errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	std::ifstream srcFile("assignment.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading assignment.cl");

	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char *src = srcProg.c_str();
	size_t length = srcProg.length();

	deviceIDs = NULL;
#if PRINT_DEBUG
	DisplayPlatformInfo(
		platformIDs[platform],
		CL_PLATFORM_VENDOR,
		"CL_PLATFORM_VENDOR");
#endif

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
			0};

	context = clCreateContext(
		contextProperties,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateContext");

	// Create program from source
	program = clCreateProgramWithSource(
		context,
		1,
		&src,
		&length,
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		"-I.",
		NULL,
		NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(
			program,
			deviceIDs[0],
			CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog),
			buildLog,
			NULL);

		std::cerr << "Error in OpenCL C source: " << std::endl;
		std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
	}

	// create buffers and sub-buffers
	inputOutput = new int[NUM_BUFFER_ELEMENTS * numDevices];
	for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
	{
		inputOutput[i] = i;
	}

	// create a single buffer to cover all the input data
	cl_mem main_buffer = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer");

	// create a 2x2 sub-buffer for every point
	for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
	{
		cl_buffer_region region =
			{
				i * sizeof(int),
				4 * sizeof(int)};
		cl_mem buffer = clCreateSubBuffer(
			main_buffer,
			CL_MEM_READ_WRITE,
			CL_BUFFER_CREATE_TYPE_REGION,
			&region,
			&errNum);
		checkErr(errNum, "clCreateSubBuffer");
		buffers.push_back(buffer);
	}

	// create a command queue for every sub-buffer
	for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
	{
		cl_command_queue queue =
			clCreateCommandQueue(
				context,
				deviceIDs[0],
				0,
				&errNum);
		checkErr(errNum, "clCreateCommandQueue");

		queues.push_back(queue);

		cl_kernel kernel = clCreateKernel(
			program,
			"filter",
			&errNum);
		checkErr(errNum, "clCreateKernel(filter)");

		errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
		checkErr(errNum, "clSetKernelArg(filter)");

		kernels.push_back(kernel);
	}

	std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
	// Perform a number of iterations
	for (size_t iter = 0; iter < iterations; iter++)
	{
		// Write input data
		errNum = clEnqueueWriteBuffer(
			queues[numDevices - 1],
			main_buffer,
			CL_TRUE,
			0,
			sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
			(void *)inputOutput,
			0,
			NULL,
			NULL);

		std::vector<cl_event> events;
		// call kernel for each device
		for (unsigned int i = 0; i < queues.size(); i++)
		{
			cl_event event;

			const size_t globalWorkSize[2] = {4, 4};
			const size_t localWorkSize[2] = {2, 2};

			errNum = clEnqueueNDRangeKernel(
				queues[i],
				kernels[i],
				2,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				0,
				&event);

			events.push_back(event);
		}

		// Technically don't need this as we are doing a blocking read
		// with in-order queue.
		clWaitForEvents(events.size(), &events[0]);
	}
	std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();
	std::cout << iterations << ","
		<< std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()
		<< std::endl;

	// Read back computed data
	clEnqueueReadBuffer(
		queues[numDevices - 1],
		main_buffer,
		CL_TRUE,
		0,
		sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
		(void *)inputOutput,
		0,
		NULL,
		NULL);

#if PRINT_DEBUG
	// Display output in rows
	for (unsigned i = 0; i < numDevices; i++)
	{
		for (unsigned elems = i * NUM_BUFFER_ELEMENTS; elems < ((i + 1) * NUM_BUFFER_ELEMENTS); elems++)
		{
			if (elems % 4 == 0 && elems > 0)
				std::cout << std::endl;
			std::cout << " " << inputOutput[elems];
		}
		std::cout << std::endl;
	}
	std::cout << "Program completed successfully" << std::endl;
#endif

	return 0;
}
