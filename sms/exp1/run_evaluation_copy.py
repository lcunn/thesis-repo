import logging
import yaml
import argparse
import os
from pathlib import Path
import torch
import pickle as pkl
import numpy as np
from uuid import uuid4
from typing import List, Dict

from sms.src.log import configure_logging
from sms.src.vector_search.evaluate_top_k import create_augmented_data, build_model, create_embedding_dict, embeddings_to_faiss_index, evaluate_top_k, evaluate_search

from pydantic import BaseModel
from sms.exp1.config_classes import LaunchPlanConfig, load_config_from_launchplan

np.random.seed(42)
torch.manual_seed(42)

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

        # Load pre-computed embeddings
        embeddings_file = eval_config.mod_path.replace('.pth', '_embeddings.pt')
        embeddings_dict = torch.load(embeddings_file)
        logger.info(f"Loaded embedding dictionary for {len(embeddings_dict)} keys.")

        # create augmented embeddings structure
        augmented_embeddings_dict = {}
        for data_id, aug_dict in augmented_data.items():
            dumped_lp_config = eval_config.lp_config.model_dump()
            bm_cfg = {'full_model_path': eval_config.mod_path} if eval_config.path_type == 'full' else {'encoder_path': eval_config.mod_path}
            model = build_model(dumped_lp_config, **bm_cfg, use_full_model=eval_config.use_full_model)
            augmented_embeddings_dict[data_id] = create_embedding_dict(aug_dict, dumped_lp_config, model)
        logger.info(f"Created augmented embeddings.")

        dim = list(embeddings_dict.values())[0].shape[0]
        index = embeddings_to_faiss_index(embeddings_dict=embeddings_dict, index_type="IndexFlatL2", index_args=[dim])
        logger.info(f"Created FAISS index.")
        
        results[eval_config.name] = evaluate_search(embeddings_dict, augmented_embeddings_dict, [1, 3, 5, 10, 25, 50, 100], index)
        logger.info(f"Evaluated top K.")
    return results

def eval_exp1_runs():
    # Load the pre-saved data_dict
    data_dict = torch.load(r"data/exp1/val_data_dict.pt")
    logger.info(f"Loaded data_dict with {len(data_dict)} entries.")

    runs_dir = Path("plagdet/sms/exp1/runs")
    
    for run_folder in runs_dir.iterdir():
        if not run_folder.is_dir():
            continue
        
        eval_folder = run_folder / "eval"
        eval_folder.mkdir(exist_ok=True)
        
        lp_config = load_config_from_launchplan(run_folder / "original_launchplan.yaml")
        
        model_configs = []
        
        # Pretrain models
        pretrain_models = [
            ("pretrain_saved_model.pth", "pretrain_eval.pt"),
            ("pretrain_saved_model_last.pth", "pretrain_eval_last.pt")
        ]
        for model_file, eval_file in pretrain_models:
            model_path = run_folder / model_file
            if model_path.exists():
                eval_file_path = eval_folder / eval_file
                embeddings_file = run_folder / "embeddings" / f"{run_folder.name}_{model_file.replace('.pth', '_embeddings.pt')}"
                logger.info(f"Checking for embeddings file: {embeddings_file}")
                if not eval_file_path.exists() and embeddings_file.exists():
                    model_configs.append(ModelEvalConfig(
                        name=f"{run_folder.name}_{model_file.split('.')[0]}",
                        lp_config=lp_config,
                        mod_path=str(model_path),
                        path_type='full',
                        use_full_model=True
                    ))
        
        # Finetune models
        finetune_models = [
            ("finetune_saved_model.pth", "finetune_eval.pt"),
            ("finetune_saved_model_last.pth", "finetune_eval_last.pt")
        ]
        for model_file, eval_file in finetune_models:
            model_path = run_folder / model_file
            if model_path.exists():
                eval_file_path = eval_folder / eval_file
                embeddings_file = run_folder / "embeddings" / f"{run_folder.name}_{model_file.replace('.pth', '_embeddings.pt')}"
                logger.info(f"Checking for embeddings file: {embeddings_file}")
                if not eval_file_path.exists() and embeddings_file.exists():
                    model_configs.append(ModelEvalConfig(
                        name=f"{run_folder.name}_{model_file.split('.')[0]}",
                        lp_config=lp_config,
                        mod_path=str(model_path),
                        path_type='encoder',
                        use_full_model=False
                    ))
        
        # Run evaluation for the models that haven't been evaluated yet
        if model_configs:
            logger.info(f"Running evaluations for {run_folder.name}")
            results = run_evaluation(data_dict, 1000, model_configs)
            
            # Save results
            for config in model_configs:
                if config.path_type == 'full':
                    eval_file = "pretrain_eval.pt" if "last" not in config.name else "pretrain_eval_last.pt"
                else:
                    eval_file = "finetune_eval.pt" if "last" not in config.name else "finetune_eval_last.pt"
                torch.save(results[config.name], eval_folder / eval_file)
                logger.info(f"Saved evaluation results for {config.name} to {eval_folder / eval_file}")
        else:
            logger.info(f"No models to evaluate for {run_folder.name}")
        
        logger.info(f"Completed evaluation for {run_folder.name}")

    logger.info("All evaluations completed.")

if __name__ == "__main__":
    eval_exp1_runs()