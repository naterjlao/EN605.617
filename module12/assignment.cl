//-----------------------------------------------------------------------------
/// @file assignment.cu
/// @author Nate Lao (nlao1@jh.edu)
/// @brief OpenCL Module12 Kernel Function
//-----------------------------------------------------------------------------
__kernel void filter(__global * buffer)
{
	const size_t y = get_local_id(0);
	const size_t x = get_local_id(1);

	int average = 0;

	// Get current position
	average += buffer[y*4 + x];

	// Get east neighbor
	const size_t dx = (x + 1);
	average += buffer[y*4 + dx];

	// Get south neighbor
	const size_t dy = (y + 1);
	average += buffer[dy*4 + x];

	// Get south-east neighbor
	average += buffer[dx*4 + dx];

	// Compute average
	average = average / 4;
	buffer[y*4 + x] = average;
}