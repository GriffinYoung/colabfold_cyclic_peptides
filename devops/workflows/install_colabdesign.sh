#!/bin/bash

$SCHRODINGER/run schrodinger_virtualenv.py venv
source venv/bin/activate
python3 -m pip -qy install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1

# Try copying existing environment
#gsutil -m cp -r gs://alphafold-environments/cyclic_peptide_design_environment/ params
if ! [ -d params ]; then
    mkdir params
    wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
    tar -xf alphafold_params_2022-12-06.tar -C params
    # Save the environment
    #gsutil -m cp -r params gs://alphafold-environments/cyclic_peptide_design_environment
fi