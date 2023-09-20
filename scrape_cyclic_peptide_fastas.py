import pandas as pd
import requests

def get_sequence(pdb_id):
    url = 'https://www.rcsb.org/fasta/entry/' + pdb_id
    response = requests.get(url)
    if not response.ok:
        raise ValueError(f'Could not download fasta for {pdb_id}')
    fasta_str = response.text
    sequence = fasta_str.split('\n')[1]
    return sequence

df = pd.read_csv('supplementary_table_1.csv')
pdb_ids = df['PDB'].tolist()

fasta_str = ''
for pdb_id in pdb_ids:
    sequence = get_sequence(pdb_id)
    fasta_str += f'>{pdb_id}\n{sequence}\n'

with open('cyclic_peptides.fasta', 'w') as f:
    f.write(fasta_str)