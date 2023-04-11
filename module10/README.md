# Module 9 OpenCL

## Building
Run `make` to compile the `main` executable.

## Usage
The following syntax is used to execute `main`.
```
./main <ARRAY_SIZE>
```
Where:
- `ARRAY_SIZE`: Size of the input arrays

The time measurements for the OpenCL math operations are printed in this order:
ARRAY_SIZE,ADD TIME (us),SUB TIME (us), MUL TIME (us), POW TIME (us)

## Validation and Metrics
The `run.sh` will run compile the `main` binaries and run trial tests.
Refer the script headers for more information