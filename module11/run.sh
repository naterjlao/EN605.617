#!/bin/bash
# Usage:
# ./run.sh
make c
make main
rm -rvf results.csv
touch results.csv
echo -n "array_size,add_time_us,sub_time_us,mul_time_us,pow_time_us" >> results.csv
echo "" >> results.csv

max_size=90000
array_size=1000
while [ $array_size -le $max_size ]
do
    echo "EXECUTING $array_size ARRAY SIZE"
    ./main $array_size >> results.csv
    array_size=$(($array_size + 1000))
done