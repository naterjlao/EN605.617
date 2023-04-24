#!/bin/bash
# Usage:
# ./run.sh
make c
make main
rm -rvf results.csv
touch results.csv
echo -n "iterations,time_us" >> results.csv
echo "" >> results.csv

max_iterations=30000
iterations=100
while [ $iterations -le $max_iterations ]
do
    echo "EXECUTING $iterations ITERATIONS"
    ./main $iterations >> results.csv
    iterations=$(($iterations + 100))
done