#!/bin/bash

# Activate the conda environment
conda activate sms

# Run the training script with three different launch plans
python sms/exp1/run_training.py --lp sms/exp1/launchplans/transformer_abs.yaml --rf sms/exp1/results/transformer_abs_1
python sms/exp1/run_training.py --lp sms/exp1/launchplans/transformer_rel.yaml --rf sms/exp1/results/transformer_rel_1
python sms/exp1/run_training.py --lp sms/exp1/launchplans/transformer_quant_abs.yaml --rf sms/exp1/results/transformer_quant_abs_1