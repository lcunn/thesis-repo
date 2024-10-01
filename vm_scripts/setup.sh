#!/bin/bash

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh

# Add conda to path
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# clone the repo
git clone https://github.com/lcunn/plagdet.git

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the new environment
conda activate sms

echo "Conda environment 'sms' has been created and activated."