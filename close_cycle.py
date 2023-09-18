from schrodinger.structure import StructureReader, Structure
from schrodinger.application import prepwizard
import os
import argparse

def close_cycle(st: Structure):
    binder = list(st.chain)[0]
    n_term = list(binder.residue)[0]
    c_term = list(binder.residue)[-1]
    c_term.getCarbonylCarbon().addBond(n_term.getBackboneNitrogen(), 1)
    return st


def main():
    parser = argparse.ArgumentParser(description='Connect the ends of a pepride chain')
    parser.add_argument('structure_file',
                        type=str,
                        help='Structure to cyclicize and prepwizard')
    parser.add_argument('out_dir',
                        type=str,
                        help='Directory to save results in')
    

    args = parser.parse_args()
    st = StructureReader.read(args.structure_file)
    st = close_cycle(st)
    out_file = args.structure_file.split('.')[0] + '_closed.mae'
    st.write(out_file)
    options = {}#prepwizard.PrepWizardSettings()
    st = prepwizard.prepare_structure(st, options)[0]
    out_file = args.structure_file.split('.')[0] + '_closed_prepped.mae'
    with os.mkdir(args.out_dir):
        st.write(os.path.join(args.out_dir, out_file))

if __name__ == "__main__":
    main()
