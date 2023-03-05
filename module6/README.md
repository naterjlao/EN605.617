# Module 5 Constant vs Shared Memory

## Building
Run `make` to compile the `main` executable.

## Usage
The following syntax is used to execute `main`.
```
./main <THREADS> <BLOCK_SIZE>
```
Where:
- `EXE`: Either `pageable` or `pinned`
- `THREADS`: Name of the input text file
- `BLOCK_SIZE`: Offset value for the Caesar Cypher

The time measurements for Constant Memory and Shared Memory operations are
printed (in order) to stdout.

## Validation and Metrics
The `run.sh` will run compile the `main` binaries and run trial tests.
Refer the script headers for more information