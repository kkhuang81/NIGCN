# Effective and Scalable Graph Neural Networks for Semi-supervised Classification

This repository contains the codes of GTR and GTRO, data process, and split generation codes.

## Requirements
- CUDA 10.1.243
- python 3.6.10
- pytorch 1.4.0
- GCC 5.4.0
- [cnpy](https://github.com/rogersce/cnpy)
- [swig-4.0.1](https://github.com/swig/swig)

## Datasets

The tested datasets can be downloaded from the corresponding citation in the papers.

In particular, folder ``preprocessing`` contains the codes for data processing and split generation.

## Compilation
```sh
make
```
## Running the code

Folder ``run_scripts`` contains the script to run GTO and GTRO for each datasets. 

By default, the codes run as GTRO.  

Adding ``--no-opt`` option in command would disable the optimization, thus run as GTR. 
