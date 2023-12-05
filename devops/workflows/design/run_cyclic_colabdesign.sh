#!/bin/bash
protocol=$1
num_seqs=$2

if [[ -s "structures.maegz" ]]; then
    python3 -u design.py $protocol raw_results --num_seqs $num_seqs --design_parameters design_parameters.csv --structure_file structures.maegz
else
    python3 -u design.py $protocol raw_results --num_seqs $num_seqs --design_parameters design_parameters.csv
fi