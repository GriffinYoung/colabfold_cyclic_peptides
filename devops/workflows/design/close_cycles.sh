#!/bin/bash

mkdir -p results
python close_cycles.py raw_results results --subdirs
cp raw_results/*.sequence results

tar -czvf results.tgz results