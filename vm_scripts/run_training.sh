#!/bin/bash

# Activate the conda environment
conda activate sms

# Run the training script with three different launch plans
python sms/exp1/run_training.py --launchplan launchplan1.yaml
python sms/exp1/run_training.py --launchplan launchplan2.yaml
python sms/exp1/run_training.py --launchplan launchplan3.yaml