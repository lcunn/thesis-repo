import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os
import psutil
import time

from sms.src.log import configure_logging
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
        time_queries=True,
        measure_memory=False  # Already tracked memory
    )
    
    # Add metrics to results
    results['index_metrics'] = {
        'index_size_MB': index_size,
        'memory_used_MB': mem_used
    }
    
    return results

def main():
    embeddings_dir = Path("data/exp2/precomputed_embeddings")
    output_dir = Path("data/exp2/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    rel_cfg = ModelEvalConfig(
        name="transformer_rel_1_pretrain",
        lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_rel_1/original_launchplan.yaml"),
        mod_path="sms/exp1/runs/transformer_rel_1/pretrain_saved_model.pth",
        embeddings_path="data/exp2/precomputed_embeddings/transformer_rel_1_pretrain_embeddings.pt",
        path_type='full',
        use_full_model=True
    )

    pr_cfg = ModelEvalConfig(
        name="transformer_pr_1_pretrain",
        lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_pr_1/original_launchplan.yaml"),
        mod_path="sms/exp1/runs/transformer_pr_1/pretrain_saved_model.pth",
        embeddings_path="data/exp2/precomputed_embeddings/transformer_pr_1_pretrain_embeddings.pt",
        path_type='full',
        use_full_model=True
    )

if __name__ == "__main__":
    main()