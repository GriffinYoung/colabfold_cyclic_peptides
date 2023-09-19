mkdir -p raw_results
colabfold_batch sequence.fasta raw_results

tar -czvf raw_results.tgz raw_results

mkdir -p results
$SCHRODINGER/run close_cycles.py raw_results results

tar -czvf results.tgz results