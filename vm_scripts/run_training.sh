source activate vm-sms

python -m sms.exp1.run_training --lp sms/exp1/launchplans/transformer_rel_big.yaml --rf sms/exp1/runs/transformer_rel_big_1
python -m sms.exp1.run_training --lp sms/exp1/launchplans/transformer_quant_rel_big.yaml --rf sms/exp1/runs/transformer_quant_rel_big_1