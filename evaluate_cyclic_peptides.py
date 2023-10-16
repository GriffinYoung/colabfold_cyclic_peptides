from schrodinger import structure
from schrodinger.structutils import rmsd
from schrodinger.structutils.rmsd import cealign

from schrodinger.structutils import structalign

import os
from collections import defaultdict
from typing import List, Optional
from Bio.PDB import PDBList
import subprocess
import argparse
import matplotlib.pyplot as plt


def get_rmsd(st_ref, st_pred):
    return cealign(st_ref, st_pred, window_size=2, max_gap=0).rmsd


def get_pdb_st(pdb_id: str):
    pdb_list = PDBList()
    fname = pdb_list.retrieve_pdb_file(pdb_id, pdir=".", file_format="mmCif")

    st = structure.StructureReader.read(fname)
    os.remove(fname)
    return st


def plot_histogram(rmsd_values):
    plt.figure()
    plt.hist(rmsd_values, bins=20, edgecolor='black')
    plt.title('Combined RMSD Distribution')
    plt.xlabel('RMSD from native (Angstroms)')
    plt.ylabel('Frequency')
    plt.savefig('rmsd_histogram.png')


def main():
    parser = argparse.ArgumentParser(
        description='Connect the ends of a pepride chain')
    parser.add_argument(
        'input_dir',
        type=str,
        help='Colabfold output dir to get raw peptide structure from')
    args = parser.parse_args()

    rmsd_dict = defaultdict(list)
    raw_results_dir = args.input_dir
    for fname in os.listdir(raw_results_dir):
        if not fname.endswith('.pdb'):
            continue
        print(fname)
        pdb_id = fname.split('_unrelaxed')[0]
        ref_st = get_pdb_st(pdb_id)
        af_st = structure.StructureReader.read(
            os.path.join(raw_results_dir, fname))
        try:
            float_rmsd = get_rmsd(ref_st, af_st)
        except Exception as e:
            print(f"Skipping: {e}")
            continue
        print(float_rmsd)
        rmsd_dict[pdb_id].append(float_rmsd)

    rmsd_values = [
        rmsd for rmsd_list in rmsd_dict.values() for rmsd in rmsd_list
    ]
    plot_histogram(rmsd_values)


if __name__ == "__main__":
    main()
