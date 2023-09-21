#!/bin/bash
# create environment
conda create -n virtifier python==3.7
conda activate virtifier
# install tensorflow & co.
conda install tensorflow==1.15 numpy matplotlib pandas
