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
The `main` executable can be ran using the following command:
```
./main <X> <Y>
```

The `run.sh` script executes either main (CSV) or branch executable and outputs to a `results.csv` file.  
Usage can be found in the `run.sh` header.