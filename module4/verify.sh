#!/bin/bash
# Descriptions:
# Verifies the operation of the Caesar Cypher.
#
# Prerequisites:
# The pageable and pinned executables must be compiled prior to
# execution.
#
# Usage:
# ./verify.sh

# Run Pageable Encoding/Decoding Test
rm -rf encoded.txt
./pageable samples/beemovie.txt 5 encoded.txt

if [[ `diff samples/beemovie.txt encoded.txt` ]]
then
    echo 'PAGEABLE ENCODING TEST: PASSED'
else
    echo 'PAGEABLE ENCODING TEST: FAILED'
fi

rm -rf decoded.txt
./pageable encoded.txt -5 decoded.txt

if [[ `diff samples/beemovie.txt decoded.txt` ]]
then
    echo 'PAGEABLE DECODING TEST: FAILED'
else
    echo 'PAGEABLE DECODING TEST: PASSED'
fi

# Run Pinned Encoding/Decoding Test
rm -rf encoded.txt
./pinned samples/beemovie.txt 5 encoded.txt

if [[ `diff samples/beemovie.txt encoded.txt` ]]
then
    echo 'PINNED ENCODING TEST: PASSED'
else
    echo 'PINNED ENCODING TEST: FAILED'
fi

rm -rf decoded.txt
./pinned encoded.txt -5 decoded.txt

if [[ `diff samples/beemovie.txt decoded.txt` ]]
then
    echo 'PINNED DECODING TEST: FAILED'
else
    echo 'PINNED DECODING TEST: PASSED'
fi