#!/bin/bash

mkdir -p results
$SCHRODINGER/run close_cycles.py raw_results results

tar -czvf results.tgz results