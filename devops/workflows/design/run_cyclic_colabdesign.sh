#!/bin/bash

if [[ -e "structures.maegz" ]]; then
    python design.py $protocol raw_results --num_seqs $num_seqs --design_parameters design_parameters.csv --structure_file structures.maegz
else
    python design.py $protocol raw_results --num_seqs $num_seqs --design_parameters design_parameters.csv
fi