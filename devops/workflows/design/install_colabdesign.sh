#!/bin/bash

$SCHRODINGER/run schrodinger_virtualenv.py venv
source venv/bin/activate

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C . bin/micromamba
eval "$(./bin/micromamba shell hook --shell bash)"
micromamba create -n colabdesign jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
micromamba activate colabdesign

python3 -m pip install cmake
python -m pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1

python3 -c "import jax; print(jax.devices())"

# Try copying existing environment
#gsutil -m cp -r gs://alphafold-environments/cyclic_peptide_design_environment/ params
if ! [ -d params ]; then
    mkdir params
    wget --no-check-certificate https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
    tar -xf alphafold_params_2022-12-06.tar -C params
    # Save the environment
    #gsutil -m cp -r params gs://alphafold-environments/cyclic_peptide_design_environment
fi