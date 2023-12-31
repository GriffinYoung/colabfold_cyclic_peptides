import numpy as np
from typing import List
import argparse
import os
import re
from collections import namedtuple
import pandas as pd
import requests
import shutil
import tempfile
import contextlib


import util

from Bio import PDB
import jax
import jax.numpy as jnp
from colabdesign.af.alphafold.common import residue_constants
from colabdesign import mk_afdesign_model, clear_mem


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

# Context manager to temporarily enter a temporary directory
# and return to the original directory when done.
@contextlib.contextmanager
def tempdir():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        yield
        os.chdir(cwd)    

def save_outputs(af_model, out_fname_prefix):
    print("Saving halluciantion...")
    # save to temp dir
    with tempdir():
        af_model.save_pdb("temp.pdb")
        parser = PDB.PDBParser()
        structure = parser.get_structure('first', 'temp.pdb')
        first_model = structure[0]

    io = PDB.PDBIO()
    io.set_structure(first_model)
    io.save(f"{out_fname_prefix}.pdb")

    best_seq = af_model.get_seqs()[0]  # I think this list has only one element
    with open(f"{out_fname_prefix}.sequence", "w") as f:
        f.write(f"{best_seq}")


def fixbb(pdb_filename: str, chain: str, out_fname_prefix: str, seed: int = 0):
    """Produces a sequence and 5 alternative folds,
    each saved as a pdb file.

    Hallucinates peptide with backbone matching the input pdb file.

    :param pdb_filename: name of the pdb file to use as a backbone
    :param chain: chain to use from the pdb file
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

    save_outputs(af_model, out_fname_prefix)


def hallucination(length: int, out_fname_prefix: str, seed=0):
    """Produces a sequence and 5 alternative folds,
    each saved as a pdb file.

    Hallucinates peptide with given length.

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

    save_outputs(af_model, out_fname_prefix)


def binder(pdb_filename: str,
           chain_to_mimic: str,
           chain_to_bind: str,
           chain_to_bind_hotspot: str,
           binder_len: int,
           initial_sequence: str,
           out_fname_prefix: str,
           seed=0):
    """

    :param pdb_filename: name of the pdb file to use as a backbone
    :param chain: chain to use from the pdb file
    :param hotspot: Residues to restrict loss to, e.g. "1-10,12,15"
    :param binder_len: length of the cyclic peptide to design
    :param out_fname_prefix: prefix of the output files
    :param seed: random seed, defaults to 0
    """
    # Hallucination
    clear_mem()
    af_model = mk_afdesign_model(
        protocol="binder",
        use_multimer=True,
        num_recycles=1,
        recycle_mode="sample")  # What is recycle_mode?
    af_model.prep_inputs(
        pdb_filename=pdb_filename,
        chain=chain_to_bind,
        binder_len=binder_len,
        binder_chain=
        chain_to_mimic,  # What is binder_chain? I think it is the chain of the binder in the pdb file
        hotspot=chain_to_bind_hotspot,
        use_multimer=True,
        rm_target_seq=False
    )  # What is rm_target_seq? "allow backbone of target structure to be flexible" what does that mean?

    util.add_cyclic_offset(af_model)

    GD_method = 'adam'
    # add_rg_loss(af_model)

    af_model.restart(
        seed=seed, seq=initial_sequence
    )  # If binder_seq, pass that here and set binder_len to len(binder_seq) (must be all caps)

    af_model.set_optimizer(optimizer=GD_method,
                           learning_rate=0.1,
                           norm_seq_grad=True)  # What is norm_seq_grad?

    af_model.design_pssm_semigreedy(120,
                                    32,
                                    num_recycles=1,
                                    models=af_model._model_names,
                                    dropout=True)

    save_outputs(af_model, out_fname_prefix)


# Declare designt tuple namedtuple
Design = namedtuple('Design', [
    'structure_fname', 'jobname', 'chain_to_mimic', 'chain_to_bind',
    'chain_to_bind_hotspot', 'designed_sequence_len', 'initial_sequence'
])


def create_design_tuples(protocol: str,
                         chain_df: pd.DataFrame) -> List[Design]:
    design_tuples = []
    for pdb_id, st_title, chain_to_mimic, chain_to_bind, chain_to_bind_hotspot, designed_sequence_len, initial_sequence in chain_df.values:
        jobname = protocol
        # Load structure from file or download from PDB
        assert pdb_id is None or st_title is None
        structure_fname = None
        if pdb_id is not None:
            jobname += f"_{pdb_id}"
            structure_fname = f'{pdb_id}.pdb'
            if not os.path.exists(structure_fname):
                download_pdb(pdb_id)
        if st_title is not None:
            jobname += f"_{st_title}"
            structure_fname = f'{st_title}.pdb'

        # Designate chains to mimic and bind
        if chain_to_bind is not None:
            chain_to_bind = chain_to_bind.upper()
            jobname += f'_bind_{chain_to_bind}'
        if chain_to_mimic is not None:
            chain_to_mimic = chain_to_mimic.upper()
            jobname += f'_mimic_{chain_to_mimic}'

        # Designate sequence length, initial sequence, and hotspot
        if designed_sequence_len is not None:
            designed_sequence_len = int(designed_sequence_len)
            jobname += f'_len_{designed_sequence_len}'
        if chain_to_bind_hotspot is not None:
            resiudes_pattern = re.compile(r'(\d+|\d+-\d+,)*(\d+|\d+-\d+)+')
            assert resiudes_pattern.match(chain_to_bind_hotspot)
            jobname += f'_hotspot_{chain_to_bind_hotspot}'
        if initial_sequence is not None:
            initial_sequence = initial_sequence.upper()
            designed_sequence_len = len(initial_sequence)
            jobname += f'_init_{initial_sequence}'

        # Check that all required arguments are present
        if protocol == 'binder':
            assert structure_fname is not None
            assert chain_to_bind is not None
            assert (designed_sequence_len is not None) or (
                initial_sequence is not None) or (chain_to_mimic is not None)
        elif protocol == 'hallucination':
            assert designed_sequence_len is not None
        elif protocol == 'fixbb':
            assert structure_fname is not None
            assert chain_to_mimic is not None

        design_tuples.append(
            Design(structure_fname, jobname, chain_to_mimic, chain_to_bind,
                   chain_to_bind_hotspot, designed_sequence_len,
                   initial_sequence))
    return design_tuples


def main():
    parser = argparse.ArgumentParser(description='Design a cyclic peptide.')
    # Optional arguments
    parser.add_argument('protocol',
                        choices=['fixbb', 'hallucination', 'binder'],
                        type=str,
                        help='Design protocol to use')
    parser.add_argument('out_dir',
                        type=str,
                        help='Directory to save results in')
    parser.add_argument('--design_parameters',
                        default=None,
                        type=str,
                        help='Csv file containing design protocol parameters')

    parser.add_argument('--num_seqs',
                        default=1,
                        type=int,
                        help='Number of designs to generate per input')

    args = parser.parse_args()

    chain_df = pd.read_csv(args.design_parameters,
                           header=None,
                           delimiter=";",
                           names=[
                               'pdb_id', 'st_title', 'chain_to_mimic',
                               'chain_to_bind', 'chain_to_bind_hotspot',
                               'designed_sequence_len', 'initial_sequence'
                           ])
    chain_df = chain_df.replace(np.NaN, None)
    design_tuples = create_design_tuples(args.protocol, chain_df)

    if args.protocol == 'fixbb':
        for design in design_tuples:
            for i in range(args.num_seqs):
                out_fname_prefix = os.path.join(args.out_dir,
                                                f"fixbb_{design.jobname}_{i}")
                fixbb(design.structure_fname,
                      design.chain_to_mimic,
                      out_fname_prefix,
                      seed=i)

    elif args.protocol == 'binder':
        for design in design_tuples:
            for i in range(args.num_seqs):
                out_fname_prefix = os.path.join(
                    args.out_dir, f"binder_{design.jobname}_{i}")
                binder(design.structure_fname,
                       design.chain_to_mimic,
                       design.chain_to_bind,
                       design.chain_to_bind_hotspot,
                       design.designed_sequence_len,
                       design.initial_sequence,
                       out_fname_prefix,
                       seed=i)

    elif args.protocol == 'hallucination':
        for design in design_tuples:
            for i in range(args.num_seqs):
                out_fname_prefix = os.path.join(
                    args.out_dir, f'hallucination_{design.jobname}_{i}')
                hallucination(design.designed_sequence_len,
                              out_fname_prefix,
                              seed=i)


if __name__ == "__main__":
    main()
