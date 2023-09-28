from schrodinger.structure import StructureReader, Structure
from schrodinger.application import prepwizard
from schrodinger.structutils.measure import measure_distance
from schrodinger.protein.rotamers import Rotamers
import os
import argparse


MAX_PEPTIDE_BOND_LENGTH = 1.5
MAX_DISULFIDE_BOND_LENGTH = 3.0


def get_sulfur(res):
    return [a for a in res.atom if 'S' in a.atom_type_name][0]


def optimize_disulfide_rotamers(st, cys1, cys2):
    s1, s2 = get_sulfur(cys1), get_sulfur(cys2)
    rot1, rot2 = Rotamers(st, s1.index), Rotamers(st, s2.index)
    best_rotamer_pair = None
    best_dist = MAX_DISULFIDE_BOND_LENGTH
    for r1 in rot1.rotamers:
        r1.apply()
        for r2 in rot2.rotamers:
            r2.apply()
            dist = measure_distance(s1, s2)
            if dist <= best_dist:
                best_dist = dist
                best_rotamer_pair = (r1, r2)

    if best_rotamer_pair is None:
        raise ValueError(f'Could not find disulfide bond between {cys1} and {cys2}, shortest distance {best_dist}')
    best_rotamer_pair[0].apply()
    best_rotamer_pair[1].apply()
    s1.addBond(s2, 1)
    
def close_cycle(st: Structure):
    binder = list(st.chain)[0]
    n_term = list(binder.residue)[0]
    c_term = list(binder.residue)[-1]
    # Disulfide bonds
    if n_term.pdbres == 'CYS ' and c_term.pdbres == 'CYS ':
        optimize_disulfide_rotamers(st, n_term, c_term)
    else:
        c_carbon = c_term.getCarbonylCarbon()
        n_nitrogen = n_term.getBackboneNitrogen()
        peptide_bond_length = measure_distance(c_carbon, n_nitrogen)
        if peptide_bond_length > MAX_PEPTIDE_BOND_LENGTH:
            raise ValueError(f'Peptide bond of length {peptide_bond_length} is too long to close.')
        c_carbon.addBond(n_nitrogen, 1)
    return st


def main():
    parser = argparse.ArgumentParser(description='Connect the ends of a pepride chain')
    parser.add_argument('input_dir',
                        type=str,
                        help='Colabfold output dir to get raw peptide structure from')
    parser.add_argument('out_dir',
                        type=str,
                        help='Directory to save results in')
    
    options = prepwizard.PrepWizardSettings()
    args = parser.parse_args()
    for fname in os.listdir(args.input_dir):
        if not fname.endswith('.pdb'):
            continue
        header = fname.split('_unrelaxed')[0]
        structure_file = os.path.join(args.input_dir, fname)
        st = StructureReader.read(structure_file)
        try:
            st = close_cycle(st)
        except ValueError as e:
            print(f'Skipping {fname}: {e}')
            continue
        try:
            st = prepwizard.prepare_structure(st, options)[0]
        except:
            print(f'{fname} protein prep failed')
        jobdir = os.path.join(args.out_dir, header)
        os.makedirs(jobdir, exist_ok=True)
        prepped_out_file = fname.split('.')[0] + '_closed_prepped.mae'
        st.write(os.path.join(jobdir, prepped_out_file))
    
if __name__ == "__main__":
    main()
