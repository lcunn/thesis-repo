import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os
import psutil
import time
import faiss
from sms.src.log import configure_logging
from sms.src.vector_search.faiss_index import CustomFAISSIndex
from sms.src.vector_search.evaluate_top_k import create_augmented_data, build_model, create_embedding_dict, embeddings_to_faiss_index, evaluate_top_k, evaluate_search
from sms.exp1.config_classes import LaunchPlanConfig, load_config_from_launchplan

from pydantic import BaseModel

logger = logging.getLogger(__name__)
configure_logging()

class ModelEvalConfig(BaseModel):
    name: str
    lp_config: LaunchPlanConfig
    mod_path: str
    embeddings_path: str
    path_type: str    #'full' or 'encoder'
    use_full_model: bool

class IndexConfig(BaseModel):
    index_type: str
    index_args: List[Any] = []
    index_kwargs: Dict[str, Any] = {}

def get_memory_usage() -> float:
    """Get the current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    return mem

def run_evaluation(
    embeddings_dict: Dict[str, np.ndarray],
    augmented_embeddings_dict: Dict[str, Dict[str, np.ndarray]],
    k_list: List[int],
    index_config: IndexConfig
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    """
    Run top K and radius search for the given model and index configurations.
    """
    # Measure memory before building the index
    mem_before = get_memory_usage()
    
    # Build the FAISS index
    index = embeddings_to_faiss_index(embeddings_dict=embeddings_dict, **index_config.model_dump())
    logger.info(f"Created FAISS index of type {index_config.index_type}.")
    
    # Measure memory after building the index
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before

    index_size = index.index_size()
    
    logger.info(f"Index size: {index_size:.2f} MB, Memory used: {mem_used:.2f} MB.")
    
    # Run evaluation
    results = evaluate_search(
        embeddings_dict=embeddings_dict,
        augmented_embeddings_dict=augmented_embeddings_dict,
        k_list=k_list,
        index=index,
        time_queries=True  # Already tracked memory
    )
    
    # Add metrics to results
    results['index_metrics'] = {
        'index_size_MB': index_size,
        'memory_used_MB': mem_used
    }
    
    return results

def main():
    # load the keys
    selected_keys = torch.load(r"data/exp2/augmented_embeddings/selected_keys.pt")
    subset_keys = torch.load(r"data/exp2/augmented_embeddings/subset_keys.pt")

    output_dir = Path("sms/exp2/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # model configs
    model_configs = {}

    model_configs['rel'] = ModelEvalConfig(
        name="transformer_rel_1_pretrain",
        lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_rel_1/original_launchplan.yaml"),
        mod_path="sms/exp1/runs/transformer_rel_1/pretrain_saved_model.pth",
        aug_embeddings_paths = {
            '5k': "data/exp2/augmented_embeddings/transformer_rel_1_pretrain_aug_5k_embeddings.pt",
            '10k': "data/exp2/augmented_embeddings/transformer_rel_1_pretrain_aug_10k_embeddings.pt",
            '100k': "data/exp2/augmented_embeddings/transformer_rel_1_pretrain_aug_100k_embeddings.pt",
            '500k': "data/exp2/augmented_embeddings/transformer_rel_1_pretrain_aug_500k_embeddings.pt",
            '1m': "data/exp2/augmented_embeddings/transformer_rel_1_pretrain_aug_1m_embeddings.pt"
        },
        path_type='full',
        use_full_model=True
    )

    model_configs['pr'] = ModelEvalConfig(
        name="transformer_pr_1_pretrain",
        lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_pr_1/original_launchplan.yaml"),
        mod_path="sms/exp1/runs/transformer_pr_1/pretrain_saved_model.pth",
        aug_embeddings_paths = {
            '5k': "data/exp2/augmented_embeddings/transformer_pr_1_pretrain_aug_5k_embeddings.pt",
            '10k': "data/exp2/augmented_embeddings/transformer_pr_1_pretrain_aug_10k_embeddings.pt",
            '100k': "data/exp2/augmented_embeddings/transformer_pr_1_pretrain_aug_100k_embeddings.pt",
            '500k': "data/exp2/augmented_embeddings/transformer_pr_1_pretrain_aug_500k_embeddings.pt",
            '1m': "data/exp2/augmented_embeddings/transformer_pr_1_pretrain_aug_1m_embeddings.pt"
        },
        path_type='full',
        use_full_model=True
    )

    dim = 64 # both models are dim 64
    
    def initialise_index(
            index_type: str, 
            index_args: List[Any], 
            index_kwargs: Dict[str, Any],
            embeddings_dict: Dict[str, np.ndarray]
            ) -> CustomFAISSIndex:
        """
        Initialise the index based on the index type.
        """
        if index_type == "IndexIVFFlat":
            # only input with quantizer and dims
            index_args.append(int(np.sqrt(len(embeddings_dict))))
        return embeddings_to_faiss_index(embeddings_dict, index_type, index_args, index_kwargs)
    
    baseline_config = IndexConfig(
            name="IndexFlatL2",
            index_type="IndexFlatL2",
            index_args=[dim],
            index_kwargs={}
        )

    index_configs = [
        IndexConfig(
            name="IndexIVFFlat",
            index_type="IndexIVFFlat",
            index_args=[faiss.IndexFlatL2(dim), dim],
            index_kwargs={}
        ),
        IndexConfig(
            name="IndexPQ",
            index_type="IndexPQ",
            index_args=[dim, 8, 8],  # M=8, nbits=8
            index_kwargs={}
        ),
        IndexConfig(
            name="IndexHNSWFlat",
            index_type="IndexHNSWFlat",
            index_args=[dim, 32],  # M=32
            index_kwargs={}
        ),
        IndexConfig(
            name="IndexLSH",
            index_type="IndexLSH",
            index_args=[dim, 64],  # nbits=64
            index_kwargs={}
        )
    ]

    dataset_sizes = ['5k', '10k', '100k', '500k', '1m']
    
    # iterate over dataset sizes
    for size in dataset_sizes:
        logger.info(f"Starting evaluations for dataset size: {size}")
        
        # Retrieve the selected keys for the current dataset size
        aug_keys = selected_keys.get(size)
        sub_keys = subset_keys.get(size)
        
        # iterate over each model configuration
        for model_type in model_configs.keys():
            # load embedding and augmented embedding dicts
            if model_type == 'rel':
                embeddings_dict = torch.load(r"data/exp2/embeddings/transformer_rel_1_pretrain_embeddings_0.pt")
                embeddings_dict.extend(torch.load(r"data/exp2/embeddings/transformer_rel_1_pretrain_embeddings_1.pt"))
            else:
                embeddings_dict = torch.load(r"data/exp2/embeddings/transformer_pr_1_pretrain_embeddings_0.pt")
                embeddings_dict.extend(torch.load(r"data/exp2/embeddings/transformer_pr_1_pretrain_embeddings_1.pt"))

            embeddings_dict = {key: embeddings_dict[key] for key in sub_keys}
            augmented_embeddings_nested_dict = torch.load(model_configs[model_type]['aug_embeddings_paths'][size])
                        
            # iterate over each index configuration
            for index_config in index_configs:
                logger.info(f"Evaluating index: {index_config.name} for model: {model_type} and dataset size: {size}")
                
                # Run evaluation
                try:
                    results = run_evaluation(embeddings_dict, augmented_embeddings_nested_dict, [1, 3, 5, 10, 25, 50, 100], index_config)
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    continue
                
                if not results:
                    logger.error(f"No results obtained for index {index_config.name} on model {model_type} with dataset size {size}.")
                    continue
                
                # Prepare result data
                result_data = {
                    'model': model_type,
                    'dataset_size': size,
                    'index': index_config.name,
                    'index_type': index_config.index_type,
                    'index_params': {
                        'args': index_config.index_args,
                        'kwargs': index_config.index_kwargs
                    },
                    'metrics': results
                }
                
                # Define the output path
                result_file = output_dir / f"{model_type}_{index_config.name}_{size}_results.pt"
                try:
                    torch.save(result_data, result_file)
                    logger.info(f"Saved results to {result_file}")
                except Exception as e:
                    logger.error(f"Error saving results to {result_file}: {e}")
        
        logger.info(f"Completed evaluations for dataset size: {size}")
    
    logger.info("All evaluations completed.")

if __name__ == "__main__":
    main()