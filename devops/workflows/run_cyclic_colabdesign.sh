#!/bin/bash

$SCHRODINGER/run schrodinger_virtualenv.py venv
source venv/bin/activate
python -m pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1

# Try copying existing environment
#gsutil -m cp -r gs://alphafold-environments/cyclic_peptide_design_environment/ params
if ! [ -d params ]; then
    mkdir params
    wget --no-check-certificate https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
    tar -xf alphafold_params_2022-12-06.tar -C params
    # Save the environment
    #gsutil -m cp -r params gs://alphafold-environments/cyclic_peptide_design_environment
fi

mkdir -p raw_results
if [[ -e "structures.maegz" ]]; then
    python design.py $protocol raw_results --num_seqs $num_seqs --design_params design_params --structure_file structures.maegz
else
    python design.py $protocol raw_results --num_seqs $num_seqs --design_params design_params
fi

mkdir -p results
python close_cycles.py raw_results results --subdirs
cp raw_results/*.sequence results

tar -czvf raw_results.tgz raw_results
tar -czvf results.tgz results