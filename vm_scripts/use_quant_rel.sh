source activate vm-sms

# exp2

# need to create aug embeddings
# need to get normal embeddings

python -m sms.exp2.data.create_augmented_embedding_datasets
python -m sms.exp2.data.create_embedding_datasets

# exp3

python -m sms.exp3.data.embed_chunks

# run exp2 

python -m sms.exp2.run_experiment