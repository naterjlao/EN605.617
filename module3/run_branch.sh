#!/bin/bash
# Usage:
# ./run_csv.sh <n> [v] [b]
# Where:
# n - number of iterations
# v - vector size (optional)
# b - block size (optional)
# Example:
# ./run_csv.sh csv 10
make c
make branch
rm -rvf results.csv
touch results.csv
echo -n "data_size,num_blocks,block_size," >> results.csv
echo -n "gpu,cpu" >> results.csv
echo "" >> results.csv

for (( i=1; i <= $1; i++ ))
do
    threads=$((1048576 + $i*65792))
    echo "threads=$threads"
    ./branch $threads >> results.csv
done