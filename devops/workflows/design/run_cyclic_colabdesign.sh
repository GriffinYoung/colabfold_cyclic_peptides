#!/bin/bash
protocol=$1
num_seqs=$2

mkdir pdb_dir

if [[ -s "structures.maegz" ]]; then
    $SCHRODINGER/run python3 -u split_maegz.py structures.maegz pdb_dir
else
    python3 -u design.py $protocol raw_results --num_seqs $num_seqs --design_parameters design_parameters.csv --pdb_dir pdb_dir
fi