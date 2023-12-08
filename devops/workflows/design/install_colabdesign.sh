#!/bin/bash

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C . bin/micromamba
eval "$(./bin/micromamba shell hook --shell bash)"
# If conda env doesn't exist
ENV_NAME="colabdesign"
if micromamba env list | grep -q "$ENV_NAME"; then
    echo "$ENV_NAME environment already exists."
else
    micromamba create -n $ENV_NAME jaxlib=*=*cuda* jax cuda-nvcc requests -c conda-forge -c nvidia
    micromamba activate $ENV_NAME

    python3 -m pip install cmake
    python3 -m pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
fi

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