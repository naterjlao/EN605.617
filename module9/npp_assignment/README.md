# Module 9 NPP Library

## Building
Run `make` to compile the `main` executable.

## Usage
The following syntax is used to execute `main`.
```
./main <DIMENSION_SIZE> <ITERATIONS>
```
Where:
- `DIMENSION_SIZE`: Size of the input dimension matrices
- `ITERATIONS`: Number of Iterations to perform

The time measurements for NPP Min-Max Operations are printed in this order:
DIMENSION_SIZE,ITERATIONS,TIME (in milliseconds)

## Validation and Metrics
The `run.sh` will run compile the `main` binaries and run trial tests.
Refer the script headers for more information