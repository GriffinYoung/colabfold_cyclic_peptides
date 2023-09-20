export COLABFOLDDIR="localcolabfold"
#mkdir -p $COLABFOLDDIR
# Try copying existing environment
#gsutil -m cp -r gs://alphafold-environments/cyclic_peptide_environment/* $COLABFOLDDIR
if ! [ -d $COLABFOLDDIR ]; then
    # Install colabfold
    apt-get update && apt-get -y install curl wget git
    apt-get -y install curl # for install_colabbatch_linux.sh
    wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh && source install_colabbatch_linux.sh
    conda activate $COLABFOLDDIR/colabfold-conda
    conda install -c nvidia cuda-nvcc

    # Modify colabfold to handle cyclic peptides
    export SITEPACKAGES="${COLABFOLDDIR}/colabfold-conda/lib/python3.10/site-packages/"
    printf "\ndef add_cyclic_offset(offset, L):\n  max_dist = (L // 2)\n  idxs = jnp.arange(offset.shape[0])\n  i = idxs[:,None]\n  j = idxs[None,:]\n  set_indices = (i < L) & (j < L)\n  dists = abs(j - i)\n  dists = jnp.where((dists > max_dist), L - dists, dists)\n  upper_right = (i < j)\n  offset = jnp.where(set_indices & upper_right, -dists, offset)\n  offset = jnp.where(set_indices & ~upper_right, dists, offset)\n  return offset" >> $SITEPACKAGES/alphafold/model/utils.py
    sed -i "s/offset = pos\[:, None\] - pos\[None, :\]/offset = pos[:, None] - pos[None, :]\n    binder_len=jnp.sum(asym_id == 0)\n    offset = utils.add_cyclic_offset(offset,binder_len)\n/g" $SITEPACKAGES/alphafold/model/modules_multimer.py
    sed -i "s/offset = pos\[:,None\] - pos\[None,:\]/offset = pos[:,None] - pos[None,:]\n        offset = utils.add_cyclic_offset(offset,len(pos))\n/g" $SITEPACKAGES/alphafold/model/modules.py

    # Save the environment
    #gsutil -m cp -r $COLABFOLDDIR gs://alphafold-environments/cyclic_peptide_environment
fi

# Add to path so colabfold_batch can be called
export PATH="${COLABFOLDDIR}/colabfold-conda/bin:$PATH"



