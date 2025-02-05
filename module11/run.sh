#!/bin/bash
# Usage:
# ./run.sh
make c
make main
rm -rvf results.csv
touch results.csv
echo -n "iterations,time_us" >> results.csv
echo "" >> results.csv

max_iterations=100000
iterations=1000
while [ $iterations -le $max_iterations ]
do
    echo "EXECUTING $iterations ITERATIONS"
    ./main $iterations >> results.csv
    iterations=$(($iterations + 1000))
done