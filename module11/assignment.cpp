//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

#define PRINT_DEBUG 0

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalDim = 49;
cl_float inputSignal[inputSignalDim][inputSignalDim];

const unsigned int outputSignalDim = 43;
cl_float outputSignal[outputSignalDim][outputSignalDim];

const unsigned int kernelMatrixDim = 7;
cl_float kernelMatrix[kernelMatrixDim][kernelMatrixDim] =
	{
		{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
		{0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25},
		{0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25},
		{0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25},
		{0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25},
		{0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25},
		{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
};

///
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

void CL_CALLBACK contextCallback(
	const char *errInfo,
	const void *private_info,
	size_t cb,
	void *user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

int64_t ExecuteIterations(
	const size_t iterations,
	cl_command_queue queue,
	cl_kernel kernel,
	const size_t *globalWorkSize,
	const size_t *localWorkSize,
	cl_mem outputSignalBuffer)
{
	cl_int errNum;
	int64_t executionTime = -1;

	std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
	for (size_t iter = 0; iter < iterations; iter++)
	{
		errNum = clEnqueueNDRangeKernel(
			queue,
			kernel,
			2,
			NULL,
			globalWorkSize,
			localWorkSize,
			0,
			NULL,
			NULL);
		checkErr(errNum, "clEnqueueNDRangeKernel");

		errNum = clEnqueueReadBuffer(
			queue,
			outputSignalBuffer,
			CL_TRUE,
			0,
			sizeof(cl_uint) * outputSignalDim * outputSignalDim,
			outputSignal,
			0,
			NULL,
			NULL);
		checkErr(errNum, "clEnqueueReadBuffer");
	}
	std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();

	if (errNum == CL_SUCCESS)
    {
        executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    }

    return executionTime;
}

///
//	main() for Convoloution example
//
int main(int argc, char **argv)
{
	// User input-> number of iterations
	size_t iterations = 1;
	if (argc >= 2)
	{
		iterations = atoi(argv[1]);
	}

	// Generate Random Input Signal
	srand(time(NULL)); // seed RNG
	for (unsigned int idx = 0; idx < (inputSignalDim * inputSignalDim); idx++)
		*(((cl_float *)inputSignal) + idx) = (cl_float)(rand() / ((cl_float)RAND_MAX));

	cl_int errNum;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id *platformIDs;
	cl_device_id *deviceIDs;
	cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

	// First, select an OpenCL platform to run on.
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	platformIDs = (cl_platform_id *)alloca(
		sizeof(cl_platform_id) * numPlatforms);

	errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
			platformIDs[i],
			CL_DEVICE_TYPE_GPU,
			0,
			NULL,
			&numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
		{
			checkErr(errNum, "clGetDeviceIDs");
		}
		else if (numDevices > 0)
		{
			deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices,
				&deviceIDs[0],
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
		}
	}

	// Next, create an OpenCL context on the selected platform.
	cl_context_properties contextProperties[] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)platformIDs[i],
			0};
	context = clCreateContext(
		contextProperties,
		numDevices,
		deviceIDs,
		&contextCallback,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("FloatConvolution.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading FloatConvolution.cl");

	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char *src = srcProg.c_str();
	size_t length = srcProg.length();

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
		NULL,
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

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
	}

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"fl_convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float) * inputSignalDim * inputSignalDim,
		static_cast<void *>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float) * kernelMatrixDim * kernelMatrixDim,
		static_cast<void *>(kernelMatrix),
		&errNum);
	checkErr(errNum, "clCreateBuffer(kernelMatrix)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_float) * outputSignalDim * outputSignalDim,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalDim);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &kernelMatrixDim);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = {outputSignalDim, outputSignalDim};
	const size_t localWorkSize[2] = {1, 1};

	std::cout << iterations << ","
		<< ExecuteIterations(iterations, queue, kernel, globalWorkSize, localWorkSize, outputSignalBuffer) // microseconds
		<< std::endl;

#if PRINT_DEBUG
	// Output the result buffer
	for (int y = 0; y < outputSignalDim; y++)
	{
		for (int x = 0; x < outputSignalDim; x++)
		{
			std::cout << outputSignal[y][x] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl
			  << "Executed program succesfully." << std::endl;
#endif

	return 0;
}
