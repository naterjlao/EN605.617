#!/bin/bash
# Usage:
# ./run.sh
make c
make main
rm -rvf results.csv
touch results.csv
echo -n "threads,serial,async" >> results.csv
echo "" >> results.csv

# 2^25
maxThreads=33554432
threads=1024
while [ $threads -le $maxThreads ]
do
    echo "EXECUTING $threads THREADS"
    echo -n "$threads," >> results.csv
    ./main $threads >> results.csv
    threads=$(($threads * 2))
done