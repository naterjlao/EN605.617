# Module 12 OpenCL Sub-Buffer

## Building
Run `make` to compile the `main` executable.

## Usage
The following syntax is used to execute `main`.
```
./main <ITERATIONS>
```
Where:
- `INTERATIONS`: Number of Sub-Buffer operations to perform.

The time measurements for the OpenCL math operations are printed in this order:
ITERATIONS,EXECUTION_TIME(us)

## Validation and Metrics
The `run.sh` will run compile the `main` binaries and run trial tests.
Refer the script headers for more information