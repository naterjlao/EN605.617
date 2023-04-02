#!/bin/bash
# Usage:
# ./run.sh
make c
make main
rm -rvf results.csv
touch results.csv
echo -n "vector_size,iterations,milliseconds" >> results.csv
echo "" >> results.csv

max_size=8388608
max_iterations=10000

iterations=1000
while [ $iterations -le $max_iterations ]
do
    size=256
    while [ $size -le $max_size ]
    do
        echo "EXECUTING SIZE=$size ITERATIONS=$iterations"
        ./main $size $iterations >> results.csv
        size=$(($size * 2))
    done
    iterations=$(($iterations + 1000))
done