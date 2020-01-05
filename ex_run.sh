#!/bin/bash
set -x
python annotationtest.py -f high_data_loom.loom -e 100 -n 10 -p n

python annotationtest.py -f high_data_loom.loom -e 100 -n 5 -p n
python annotationtest.py -f high_data_loom.loom -e 100 -n 20 -p n

python annotationtest.py -f high_data_loom.loom -e 200 -n 10 -p n
python annotationtest.py -f high_data_loom.loom -e 50 -n 10 -p n

python annotationtest.py -f simulation_3.loom -e 100 -n 10 -p n
python annotationtest.py -f data_loom.loom -e 100 -n 10 -p n
