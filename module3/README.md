# Building
Three executables can be built from the makefile:
- `main` (with Normal Logging)
- `main` (with CSV Logging)
- `branch`

## Main with Normal Logging Executable
The Normal Logging `main` executable prints verbose operation logging to standard out.
Compilation command:
```
make
```

## Main with CSV Logging Executable
The CSV Logging `main` executable prints CSV friendly logging to standard out.
Compilation command (note that a `make clean` must be performed prior):
```
make c csv
```

## Branch Executable
The `branch` executable performs conditional branching testing and prints results to CSV friendly logs to standard out.
Compilation command:
```
make branch
```

# Execution
The `main` executable can be run using the following command:
```
./main <X> <Y>

Where:
- X is the number of threads
- Y is the block size
```

The `branch` executable can be run by using the following command:
```
./branch <X> <Y>

Where:
- X is the number of threads
- Y is the block size
```

## Run
The `run_csv.sh` script executes The Main (CSV) executable in sequence and outputs to a `results.csv` file.  
Usage can be found in the `run_csv.sh` header.  

The `run_branch.sh` script executes The Branch executable in sequence and outputs to a `results.csv` file.  
Usage can be found in the `run_csv.sh` header.  

