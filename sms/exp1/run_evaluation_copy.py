import logging
import yaml
import argparse
import torch
import pickle as pkl
import numpy as np
from uuid import uuid4
from typing import List, Dict

from sms.src.log import configure_logging
from sms.src.vector_search.evaluate_top_k import create_augmented_data, build_model, create_embedding_dict, embeddings_to_faiss_index, evaluate_top_k, evaluate_search

from pydantic import BaseModel
from sms.exp1.config_classes import LaunchPlanConfig, load_config_from_launchplan

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
        
        results[eval_config.name] = evaluate_search(embeddings_dict, augmented_embeddings_dict, [1, 3, 5, 10, 25, 50, 100], index)
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
    data = torch.load(r"data/exp1/val_data.pt")
    data_ids = [str(uuid4()) for _ in range(len(data))]
    data_dict = dict(zip(data_ids, data))

    trans_rel_1_full = ModelEvalConfig(
        name="trans_rel_1_full",
        lp_config=load_config_from_launchplan(r"sms/exp1/runs/transformer_rel_1/original_launchplan.yaml"),
        mod_path=r"sms/exp1/runs/transformer_rel_1/pretrain_saved_model.pth",
        path_type='full',
        use_full_model=True
    )

    results = run_evaluation(
        data_dict, 
        100, 
        [trans_rel_1_full]
    )

    torch.save(results, r"trans_rel_1_full_eval.pt")