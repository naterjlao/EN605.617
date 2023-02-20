#!/bin/bash
# Description:
# Builds and runs verify.sh and metrics.sh.
#
# Usage:
# ./run.sh [n]
# Where:
# n - number of trials to perform in metrics.sh
# Example:
# ./run.sh 10

# Compile the binaries
make

# Execute Verify
./verify.sh

# Execute Metrics.sh
./metrics.sh $1