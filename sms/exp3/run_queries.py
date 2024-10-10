import os
import numpy as np
import torch
import logging
import argparse
from typing import List, Dict, Any

from sms.src.vector_search.evaluate_top_k import (
    create_embedding_dict,
    embeddings_to_faiss_index_with_data,
    evaluate_search,
    build_model
)

from sms.src.vector_search.faiss_index import CustomFAISSIndex

# Configure Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def perform_top_k_search(
    index: CustomFAISSIndex,
    query_embeddings: Dict[str, Dict[str, np.ndarray]],
    k: int
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    
    logger.info(f"Performing Top-{k} search for plagiarism evaluation")
    results = {}
    for i, (song_id, song_dict) in enumerate(query_embeddings.items()):
        logger.info(f"Processing song {i+1}/{len(query_embeddings)}: {song_id}")
        song_results = {}
        for chunk_id, chunk_embedding in song_dict.items():
            top_k_results = index.search(chunk_embedding, k)
            song_results[chunk_id] = [
                {"id": res[0], "data": res[1], "distance": res[2]} for res in top_k_results
            ]
        results[song_id] = song_results
    logger.info("Top-K search completed")
    return results

def main(db_size: int, scales_to_use: List[float], K: int, index_type: str, index_args, index_kwargs, output_path: str):
    # Step 1: Load Data
    plag_embeddings_dict = torch.load("data/exp3/all_plag_chunks_embeddings.pt")
    maestro_embeddings_dict = torch.load("data/exp3/maestro_chunks_embeddings.pt")

    # Step 2: Ensure each case has both original and plagiarized versions
    ori_dict_embeddings_dict = {}
    plag_dict_embeddings_dict = {}

    case_numbers = set(key.split('_')[0] for key in plag_embeddings_dict.keys())
    for case in case_numbers:
        ori_key = f"{case}_ori_melody.mid"
        plag_key = f"{case}_plag_melody.mid"
        
        # check if both original and plagiarized versions exist
        if ori_key in plag_embeddings_dict and plag_key in plag_embeddings_dict:
            ori_dict_embeddings_dict[ori_key] = plag_embeddings_dict[ori_key]
            plag_dict_embeddings_dict[plag_key] = plag_embeddings_dict[plag_key]

    # Step 3: Select embeddings to use

    num_maestro_to_use = db_size - len(ori_dict_embeddings_dict)
    maestro_keys = list(maestro_embeddings_dict.keys())[:num_maestro_to_use]
    maestro_chunks_to_use = {k: v for k, v in maestro_embeddings_dict.items() if k in maestro_keys}

    all_database_embeddings = {**ori_dict_embeddings_dict, **maestro_chunks_to_use}

    # Step 4: Format embeddings for index input

    input_dict = {}

    for song_id, scale_dict in all_database_embeddings.items():
        for scale, embedding_list in scale_dict.items():
            if scale in scales_to_use:
                for i, embedding in enumerate(embedding_list):
                    input_dict[f"{song_id}_{scale}_{i}"] = (embedding, (song_id, scale, i))

    index = embeddings_to_faiss_index_with_data(input_dict, index_type, index_args, index_kwargs)

    # Step 5: Format query embeddings

    query_embeddings = {}

    for song_id, scale_dict in plag_dict_embeddings_dict.items():
        song_dict = {}
        for scale, embedding_list in scale_dict.items():
            if scale in scales_to_use:
                for i, embedding in enumerate(embedding_list):
                    song_dict[f"{song_id}_{scale}_{i}"] = embedding
        query_embeddings[song_id] = song_dict

    # Step 6: Perform Top-K Search for Plagiarism Chunks
    top_k_results = perform_top_k_search(index, query_embeddings, K)

    # Step 7: Save Top-K Results
    torch.save(top_k_results, output_path)

if __name__ == "__main__":
    db_size = 200
    scales_to_use = [1.0]
    K = 10
    dim = 64
    index_type="IndexPQ"
    index_args=[dim, 8, 8]  # nbits=64
    index_kwargs={}
    output_path = f"sms/exp3/results/top_{K}_results_db_{db_size}_index_{index_type}_singlescale.pt"
    main(db_size, scales_to_use, K, index_type, index_args, index_kwargs, output_path)

    db_size = 200
    scales_to_use = [0.5,1.0,2.0]
    K = 10
    dim = 64
    index_type="IndexPQ"
    index_args=[dim, 8, 8]  # nbits=64
    index_kwargs={}
    output_path = f"sms/exp3/results/top_{K}_results_db_{db_size}_index_{index_type}_allscales.pt"
    main(db_size, scales_to_use, K, index_type, index_args, index_kwargs, output_path)

    db_size = 1000
    scales_to_use = [1.0]
    K = 50
    dim = 64
    index_type="IndexPQ"
    index_args=[dim, 8, 8]  # nbits=64
    index_kwargs={}
    output_path = f"sms/exp3/results/top_{K}_results_db_{db_size}_index_{index_type}_singlescale.pt"
    main(db_size, scales_to_use, K, index_type, index_args, index_kwargs, output_path)

    db_size = 1000
    scales_to_use = [0.5,1.0,2.0]
    K = 50
    dim = 64
    index_type="IndexPQ"
    index_args=[dim, 8, 8]  # nbits=64
    index_kwargs={}
    output_path = f"sms/exp3/results/top_{K}_results_db_{db_size}_index_{index_type}_allscales.pt"
    main(db_size, scales_to_use, K, index_type, index_args, index_kwargs, output_path)