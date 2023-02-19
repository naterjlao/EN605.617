#!/bin/bash
# Descriptions:
# Runs n number of operations for both Pageable and Pinned Memory
# Caesar Cypher operations on samples/bigsample.txt.
# Computes the average of both types.
# Usage:
# ./metrics.sh [n]
# Where:
# n - number of trials to perform
# Example:
# ./metrics.sh 10

echo "Running Timing Metrics"
rm -rvf pageable_results.csv
rm -rvf pinned_results.csv

echo -n "Pageable Trials: "
for (( i=1; i <= $1; i++ ))
do
    echo -n "$i,"
    ./pageable samples/bigsample.txt 1 >> pageable_results.csv
done
echo ""

echo -n "Pinned Trials: "
for (( i=1; i <= $1; i++ ))
do
    echo -n "$i,"
    ./pinned samples/bigsample.txt 1 >> pinned_results.csv
done
echo ""

# Get the average
cat pageable_results.csv | awk '{sum += $0} END {print "Pageable Average =",sum/NR}'
cat pinned_results.csv | awk '{sum += $0} END {print "Pinned Average =",sum/NR}'
