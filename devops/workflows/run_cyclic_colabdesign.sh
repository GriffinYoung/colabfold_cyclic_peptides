mkdir -p raw_results
python3 design.py $protocol raw_results --hallucination_length $hallucination_length --backbone_structures backbone_structures.maegz --backbone_chains backbone_chains.txt

mkdir -p results
python3 close_cycles.py raw_results results

tar -czvf raw_results.tgz raw_results
tar -czvf results.tgz results