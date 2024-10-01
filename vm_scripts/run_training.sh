source activate vm-sms

python -m sms.exp1.run_training --lp sms/exp1/launchplans/transformer_abs.yaml --rf sms/exp1/runs/transformer_abs_1
python -m sms.exp1.run_training --lp sms/exp1/launchplans/transformer_quant_abs.yaml --rf sms/exp1/runs/transformer_quant_abs_1