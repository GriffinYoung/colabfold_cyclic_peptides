import argparse
import os
from schrodinger.structure import StructureReader
from colabdesign import mk_afdesign_model, clear_mem

import jax
import jax.numpy as jnp
from colabdesign.af.alphafold.common import residue_constants

import requests

import util


def download_pdb(pdb_id):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    response = requests.get(url)

    if response.status_code == 200:
        filename = f'{pdb_id}.pdb'
        with open(filename, 'wb') as pdb_file:
            pdb_file.write(response.content)
        return filename
    else:
        raise Exception(
            f'Error downloading PDB file for ID {pdb_id}. Status code: {response.status_code}'
        )


def add_rg_loss(self, weight=0.1):
    '''add radius of gyration loss'''

    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0]**0.365
        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight


def save_outputs(af_model, out_fname_prefix, seed):
    print("Saving halluciantion...")
    temp_pdb = out_fname_prefix + ".pdb"
    af_model.save_pdb(temp_pdb)
    sts = list(StructureReader(temp_pdb))
    os.remove(temp_pdb)

    seed_prefix = f"{out_fname_prefix}_{seed}"
    for i, st in enumerate(sts):
        design_prefix = f"{seed_prefix}_{i}"
        st.title = design_prefix
        st.write(f"{design_prefix}.pdb")

    best_seq = af_model.get_seqs()[0]  # I think this list has only one element
    with open(f"{seed_prefix}.sequence", "w") as f:
        f.write(f"{best_seq}")


def fixbb(pdb_filename, chain, out_fname_prefix, seed=0):
    """Produces 5 designs, each saved as a PDB and a sequence file
    with the prefix out_fname_prefix_{seed + i} where i is the index
    of the design in the list of 5 designs.

    :param length: length of the cyclic peptide to design
    :param out_fname_prefix: prefix of the output files
    :param seed: random seed, defaults to 0
    """
    # Fixed backbone
    clear_mem()
    af_model = mk_afdesign_model(protocol="fixbb")
    af_model.prep_inputs(pdb_filename=pdb_filename, chain=chain)
    util.add_cyclic_offset(af_model)

    af_model.restart(seed=seed)
    af_model.design_3stage()

    save_outputs(af_model, out_fname_prefix, seed)


def hallucination(length, out_fname_prefix, seed=0):
    """Produces 5 designs, each saved as a PDB and a sequence file
    with the prefix out_fname_prefix_{seed + i} where i is the index
    of the design in the list of 5 designs.

    :param length: length of the cyclic peptide to design
    :param out_fname_prefix: prefix of the output files
    :param seed: random seed, defaults to 0
    """
    # Hallucination
    clear_mem()
    af_model = mk_afdesign_model(protocol="hallucination")
    af_model.prep_inputs(length=length, rm_aa="C")
    util.add_cyclic_offset(af_model)
    # add_rg_loss(af_model)

    # pre-design with gumbel initialization and softmax activation
    af_model.restart(seed=seed)
    af_model.set_seq(mode="gumbel")
    af_model.set_opt("con",
                     binary=True,
                     cutoff=21.6875,
                     num=af_model._len,
                     seqsep=0)
    af_model.set_weights(pae=1, plddt=1, con=0.5)
    af_model.design_soft(50)
    print("Finished design soft")
    # three stage design
    af_model.set_seq(seq=af_model.aux["seq"]["pseudo"])
    af_model.design_3stage(50, 50, 10)

    save_outputs(af_model, out_fname_prefix, seed)


def main():
    parser = argparse.ArgumentParser(description='Design a cyclic peptide.')
    # Optional arguments
    parser.add_argument('protocol',
                        choices=['fixbb', 'hallucination'],
                        type=str,
                        help='Design protocol to use')
    parser.add_argument('out_dir',
                        type=str,
                        help='Directory to save results in')

    parser.add_argument(
        '--hallucination_length',
        type=int,
        help='Length of the cyclic peptide to use for hallucination protocol')
    parser.add_argument(
        '--backbone_structures',
        type=str,
        default=None,
        help='File containing backbone structures to use for fixbb protocol')
    parser.add_argument(
        '--backbone_chains',
        default=None,
        type=str,
        help='File containing PDBID_CHAIN lines to use for fixbb protocol')

    parser.add_argument('--num_seqs',
                        default=1,
                        type=int,
                        help='Number of designs to generate')

    args = parser.parse_args()

    if args.protocol == 'fixbb':
        if args.backbone_chains is not None:
            with open(args.backbone_chains) as f:
                pdb_chain_tuples = [
                    line.strip().split('_') for line in f.readlines()
                ]
            for pdb_id, chain in pdb_chain_tuples:
                pdb_filename = download_pdb(pdb_id)
                for i in range(args.num_seqs):
                    out_fname_prefix = os.path.join(args.out_dir,
                                                    f'{pdb_id}_{chain}')
                    fixbb(pdb_filename, chain, out_fname_prefix, seed=i)

        if args.backbone_structures is not None:
            for st in list(StructureReader(args.backbone_structures)):
                pdb_filename = f'{st.title}.pdb'
                st.write(pdb_filename)
                chain = list(st.chain)[0].name
                for i in range(args.num_seqs):
                    out_fname_prefix = os.path.join(args.out_dir,
                                                    f'{st.title}_{chain}')
                    fixbb(pdb_filename, chain, out_fname_prefix, seed=i)

    elif args.protocol == 'hallucination':
        for i in range(args.num_seqs):
            out_fname_prefix = os.path.join(
                args.out_dir, f'hallucination_{args.hallucination_length}')
            hallucination(args.hallucination_length, out_fname_prefix, seed=i)


if __name__ == "__main__":
    main()
