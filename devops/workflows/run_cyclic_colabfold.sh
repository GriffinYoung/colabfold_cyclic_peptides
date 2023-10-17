mkdir -p raw_results
colabfold_batch sequence.fasta raw_results

mkdir -p results
$SCHRODINGER/run close_cycles.py raw_results results --subdirs

tar -czvf raw_results.tgz raw_results
tar -czvf results.tgz results