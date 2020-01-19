This repo is the final project for CS280. 
This project implemented a semi-supervised learning extended generative model for single-cell cell type annotation.

Compared with the SCANVI model, we adopted the idea of WAE.
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

# Dataset

`data` folder contaion two of our datasets used in the experiments: `simulation_3.loom` and `high_data_loom.loom`.
 The `high_data_loom.loom` is a dataset of mouse cells from different tissues which we merged by ourselves. 
 The `simulation_3.loom` is a simulation dataset provided by [scvi](https://github.com/YosefLab/scVI).

# Structure

-annotationtest.py: the script to test annoation performance\
-dgm4sca\
|--dataset: scripys to load data\
|--inference: scripts to classify cell type by posterior inference\
|--models: scripts about generative model\
-data: folder of data files in .loom format. (simulation_3.loom, high_data_loom.loom)

# Ref

[Torch version WAE](https://github.com/schelotto/Wasserstein-AutoEncoders)

[M1+M2 Model](https://github.com/dpkingma/nips14-ssl)
