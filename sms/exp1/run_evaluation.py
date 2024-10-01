import logging
import yaml
import argparse
import torch
import pickle as pkl
import numpy as np
from typing import List, Dict

from sms.src.log import configure_logging
from sms.src.vector_search.evaluate_top_k import create_augmented_data, build_model, create_embedding_dict, embeddings_to_faiss_index, evaluate_top_k

from pydantic import BaseModel
from sms.exp1.config_classes import LaunchPlanConfig

logger = logging.getLogger(__name__)
configure_logging()

class ModelEvalConfig(BaseModel):
    name: str
    lp_config: LaunchPlanConfig
    mod_path: str
    path_type: str    #'full' or 'encoder'
    use_full_model: bool

def run_evaluation(
    data_dict: Dict[str, np.ndarray],
    num_loops: int,
    model_configs: List[ModelEvalConfig]
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:

    # generate random augmentations
    anchor_keys = np.random.choice(list(data_dict.keys()), size=num_loops, replace=False)
    augmented_data = create_augmented_data(data_dict, anchor_keys)

    results = {}
    for eval_config in model_configs:
        logger.info(f"Running evaluation for {eval_config.name}")

        dumped_lp_config = eval_config.lp_config.model_dump()
        bm_cfg = {'full_model_path': eval_config.mod_path} if eval_config.path_type == 'full' else {'encoder_path': eval_config.mod_path}

        model = build_model(dumped_lp_config, **bm_cfg, use_full_model=eval_config.use_full_model)
        embeddings_dict = create_embedding_dict(data_dict, dumped_lp_config, model)
        logger.info(f"Created embedding dictionary for {len(embeddings_dict)} keys.")
        # create augmented embeddings structure
        augmented_embeddings_dict = {}
        for data_id, aug_dict in augmented_data.items():
            augmented_embeddings_dict[data_id] = create_embedding_dict(aug_dict, dumped_lp_config, model)
        logger.info(f"Created augmented embeddings.")

        dim = list(embeddings_dict.values())[0].shape[0]
        index = embeddings_to_faiss_index(embeddings_dict=embeddings_dict, index_type="IndexFlatL2", index_args=[dim])
        logger.info(f"Created FAISS index.")
        
        results[eval_config.name] = evaluate_top_k(embeddings_dict, augmented_embeddings_dict, [1, 3, 5, 10, 25, 50, 100], index)
        logger.info(f"Evaluated top K.")
    return results

def main(data_path: str, num_loops: int, model_config_paths: List[str], output_path: str):
    data_dict = pkl.load(open(data_path, 'rb'))
    model_configs = []
    for config_path in model_config_paths:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        try:
            model_config = ModelEvalConfig(**config_data)
            model_configs.append(model_config)
        except pydantic.ValidationError as e:
            logger.error(f"Invalid configuration in {config_path}: {e}")
            raise
    results = run_evaluation(data_dict, num_loops, model_configs)
    pkl.dump(results, open(output_path, 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument('data_path', type=str, help='Path to the data file.')
    parser.add_argument('num_loops', type=int, help='Number of loops for evaluation.')
    parser.add_argument('model_config_paths', type=str, nargs='+', help='Paths to model configuration files.')
    parser.add_argument('output_path', type=str, help='Path to the output file.')
    
    args = parser.parse_args()
    main(args.data_path, args.num_loops, args.model_config_paths, args.output_path)

