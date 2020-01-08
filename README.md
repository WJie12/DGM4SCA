This repo is the final project for CS280.
The code structure is mainly inspired by [scvi](https://github.com/YosefLab/scVI)
# How to run

python annotationtest.py


|para|default|usage|
|---|---|---|
|-e|100|epochs to run|
|-f|'simualtion_3.loom'|filename of dataset|
|-n|10|labeled cell number|
|-p|'y'|weather to plot the figs|
|-t|1|times to run the experiment|

# Structure

-annotationtest.py: the script to test annoation performance\
-dgm4sca\
|--dataset: scripys to load data\
|--inference: scripts to classify cell type by posterior inference\
|--models: scripts about generative model\
-data: folder of data files in .loom format. (simulation_3.loom, data_loom.loom)



