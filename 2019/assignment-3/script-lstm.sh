#!/bin/bash

methods=( "custom" "mean" "median" "mode" "bfill" "ffill" "linear" "polynomial" )

for method in "${methods[@]}"
do
    python3 lstm.py 1 $method
done
