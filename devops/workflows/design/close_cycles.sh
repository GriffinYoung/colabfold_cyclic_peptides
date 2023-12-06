#!/bin/bash

mkdir -p results
$SCHRODINGER/run close_cycles.py raw_results results --subdirs
cp raw_results/*.sequence results

tar -czvf results.tgz results