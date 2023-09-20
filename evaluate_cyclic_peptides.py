from schrodinger import structure
from schrodinger.structutils import rmsd
from schrodinger.structutils.rmsd import cealign

from schrodinger.structutils import structalign

import os
from collections import defaultdict
from typing import List, Optional
from Bio.PDB import PDBList
import subprocess

def run_bash(args):
    print(" ".join([str(arg) for arg in args]))
    retval = subprocess.run(args, capture_output=True)
    my_stdout = retval.stdout.decode("utf-8")
    my_stderr = retval.stderr.decode("utf-8")
    print(my_stderr)
    print(my_stdout)
    success = retval.returncode == 0
    return success

def align_ligands(ref_lig: structure.Structure,
                    other_ligs: List[structure.Structure],
                    core: Optional[str] = None,
                    use_queue: bool = False):
    """
    Run flexible ligand alignment of `other_ligs` onto `ref_lig`

    For now, we will submit the alignment as one batch, but potentially
    if the number of ligands to align is very large, it would pay
    to break these jobs up into multiple batches and submit them in
    parallel.  This is part of the motivation for putting it here.

    :param ref_lig: The reference ligand structure
    :param other_ligs: The ligands to be aligned to the reference ligand
    :param core: What algorithm defines the common scaffold used fo
                    alignment, default is Bemis-Murcko
    :param use_queue: if True, we wil submit the jobs to the queue as
                        opposed to running on the driver node
    :return: List of ligands aligned to (but not including) the
                reference ligand
    """

    # Fix atoms with invalid van der waals radii from reference ligand
    # by turning into a dummy atom before aligning to avoid error
    for at in ref_lig.atom:
        if at.radius < 1.0 or at.radius > 4.0:
            at.atomic_number = -2

    in_name = 'ligand_to_align.mae'
    out_name = 'ligands_aligned.mae'

    with structure.StructureWriter(in_name) as writer:
        writer.append(ref_lig)
        writer.extend(other_ligs)

    sdgr = os.environ['SCHRODINGER']
    args = [
        os.path.join(sdgr, 'utilities', 'align_ligands'), in_name, '-o', out_name,
        '-sample', 'rapid', '-max', '1', '-close_contact', '0.5',
        '-ref', '1', '-fail_on_bad', '-verbosity', '2'
    ]

    args.append('-NOJOBID')

    success = run_bash(args)

    os.remove(in_name)
    if success:
        aligned_ligands = list(structure.StructureReader(out_name))[1:]
        os.remove(out_name)
    else:
        raise ValueError(
            f'Failed to align ligands to {ref_lig.title}. See {os.path.basename(in_name)}.log'
        )

    return aligned_ligands


def get_rmsd(st_ref, st_pred):
    return cealign.get_rmsd(st_ref, st_pred).rmsd
    st_pred = align_ligands(st_ref, [st_pred])[0]

    # Calculate the RMS against the originals:
    rms = rmsd.ConformerRmsd(st_ref, st_pred)
    return rms.calculate()

def get_pdb_st(pdb_id:str):
    pdb_list = PDBList()
    fname = pdb_list.retrieve_pdb_file(pdb_id, pdir=".", file_format="mmCif")

    st = structure.StructureReader.read(fname)
    os.remove(fname)
    return st

rmsd_dict = defaultdict(list)
raw_results_dir = "/Users/young1/Downloads/raw_results 3"
for fname in os.listdir(raw_results_dir):
    if not fname.endswith('.pdb'):
        continue
    print(fname)
    pdb_id = fname.split('_unrelaxed')[0]
    ref_st = get_pdb_st(pdb_id)
    af_st = structure.StructureReader.read(os.path.join(raw_results_dir, fname))
    try:
        float_rmsd = get_rmsd(ref_st, af_st)
    except Exception as e:
        print(f"Skipping: {e}")
        continue
    print(float_rmsd)
    rmsd_dict[pdb_id].append(float_rmsd)

import pdb;pdb.set_trace()