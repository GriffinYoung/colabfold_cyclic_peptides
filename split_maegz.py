import argparse
import os

from schrodinger import structure


def split_maegz(maegz_file, out_dir):
    """
    Split a maegz file into individual pdb files.

    Parameters
    ----------
    maegz_file : str
        Path to maegz file to split.
    out_dir : str
        Path to directory to save pdb files in.
    """
    # Read in maegz file
    reader = structure.StructureReader(maegz_file)
    # Iterate through structures and save as pdb
    for st in reader:
        print("Saving structure: ", st.title)
        st.write(os.path.join(out_dir, f'{st.title}.pdb'))


def main():
    parser = argparse.ArgumentParser(description='Design a cyclic peptide.')
    # Optional arguments
    parser.add_argument('maegz_file', type=str, help='File to split')
    parser.add_argument('out_dir',
                        type=str,
                        help='Directory to save results in')

    args = parser.parse_args()
    split_maegz(args.maegz_file, args.out_dir)


if __name__ == '__main__':
    main()
