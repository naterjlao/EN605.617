# Module 8 CUDA Libraries

## Building
Run `make` to compile the `main` executable.

## Usage
The following syntax is used to execute `main`.
```
./main <THREADS> <BLOCK_SIZE>
```
Where:
- `THREADS`: Number of Threads
- `BLOCK_SIZE`: Number of Blocks

The time measurements for Random Float Generation and Matrix Multiplication are printed in this order:
NUMBER OF ELEMENTS IN MATRIX, CPU RAND, GPU RAND, CPU MAT MULT, GPU MAT MULT

## Validation and Metrics
The `run.sh` will run compile the `main` binaries and run trial tests.
Refer the script headers for more information