#!/bin/bash

mkdir -p raw_results
if [[ -e "structures.maegz" ]]; then
    python design.py $protocol raw_results --num_seqs $num_seqs --design_parameters design_parameters.csv --structure_file structures.maegz
else
    python design.py $protocol raw_results --num_seqs $num_seqs --design_parameters design_parameters.csv
fi

mkdir -p results
python close_cycles.py raw_results results --subdirs
cp raw_results/*.sequence results

tar -czvf raw_results.tgz raw_results
tar -czvf results.tgz results