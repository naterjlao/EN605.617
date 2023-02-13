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
make csv
rm -rvf results.csv
touch results.csv
echo -n "data_size,num_blocks,block_size," >> results.csv
echo -n "gpu_add,gpu_sub,gpu_mul,gpu_mod," >> results.csv
echo -n "cpu_add,cpu_sub,cpu_mul,cpu_mod" >> results.csv
echo "" >> results.csv

for (( i=1; i <= $1; i++ ))
do
    echo "run $i"
    ./main $2 $3 >> results.csv
done