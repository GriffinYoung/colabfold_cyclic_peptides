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
if [[ "$protocol" == "fixbb" ]]; then
    echo "Running fixbb protocol"
    if [[ -e "backbone_structures.maegz" ]]; then
        echo "Using backbone structures to design $num_designs designs"
        python design.py $protocol raw_results --backbone_structures backbone_structures.maegz --num_designs $num_designs
    fi

    if [[ -e "backbone_chains.txt" ]]; then
        echo "Using backbone chains to design $num_designs designs"
        python design.py $protocol raw_results --backbone_chains backbone_chains.txt --num_designs $num_designs
    fi
else
    echo "Running hallucination protocol"
    if [[ -z "$hallucination_length" ]]; then
        echo "No hallucination length specified, skipping hallucination"
    else
        echo "Hallucinating $num_designs $hallucination_length residues"
        python design.py hallucination raw_results --hallucination_length $hallucination_length --num_designs $num_designs
    fi
fi

mkdir -p results
python close_cycles.py raw_results results
cp raw_results/*.sequence results

tar -czvf raw_results.tgz raw_results
tar -czvf results.tgz results