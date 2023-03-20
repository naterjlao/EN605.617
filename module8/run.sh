#!/bin/bash
# Usage:
# ./run.sh
make c
make main
rm -rvf results.csv
touch results.csv
echo -n "matrix_size,cpu_rand,gpu_rand,cpu_mult,gpu_mult" >> results.csv
echo "" >> results.csv

# 2^210
maxDim=1024
threads=2
while [ $threads -le $maxDim ]
do
    echo "EXECUTING $threads THREADS"
    echo -n "$threads," >> results.csv
    ./main $threads >> results.csv
    threads=$(($threads * 2))
done