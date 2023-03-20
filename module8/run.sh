#!/bin/bash
# Usage:
# ./run.sh
make c
make main
rm -rvf results.csv
touch results.csv
echo -n "matrix_size,cpu_rand,gpu_rand,cpu_mult,gpu_mult" >> results.csv
echo "" >> results.csv

maxDim=2048
threads=2
while [ $threads -le $maxDim ]
do
    echo "EXECUTING $threads x $threads DIM"
    ./main $threads >> results.csv
    threads=$(($threads * 2))
done