#!/bin/bash

# Create a new conda environment named 'ugs-ind'
# conda create -n ugs-ind python=3.8 -y

# Activate the environment
# conda activate ugs-ind

# Install PyTorch and torchvision
# conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Set the CUDA variable
CUDA=cu118

# Install PyTorch Geometric extensions

pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
# pip install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html

# Install PyTorch Geometric
conda install pyg -c pyg

echo "Installation completed!"

