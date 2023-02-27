#!/bin/bash
# Usage:
# ./run.sh <n> [v] [b]
# Where:
# n - number of iterations
# v - vector size (optional)
# b - block size (optional)
# Example:
# ./run.sh 10
make c
make main
rm -rvf results.csv
touch results.csv
echo -n "const_memory,shared_memory" >> results.csv
echo "" >> results.csv

for (( i=1; i <= $1; i++ ))
do
    echo "run $i"
    ./main $2 $3 >> results.csv
done