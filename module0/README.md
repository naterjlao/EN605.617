Build CUDA
nvcc <file>

Build OpenCL
g++ <file> -lOpenCL




Download CUDA for Debian 10.

OpenCL
See https://forums.debian.net/viewtopic.php?t=145476
Additional info needed to get this to work. libOpenCL.so was still needed.

I used apt-file to look for libOpenCL.so, and I thought I found it in two packages:
nvidia-libopencl1
ocl-icd-libopencl1

I tried installing these and running the fahclient, one at a time, but no luck.

I then realized that these packages provide:
/usr/lib/x86_64-linux-gnu/libOpenCL.so.1
not:
/usr/lib/x86_64-linux-gnu/libOpenCL.so

So I ran the following command:
sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/x86_64-linux-gnu/libOpenCL.so

to create a symlink from the filename fah was looking for, to the symlink provided by the nvidia-libopencl1 package. (Which is then finally symlinked to the actual lib)

This worked!
