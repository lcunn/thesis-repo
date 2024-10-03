import logging
import yaml
import os
from pathlib import Path
import torch
import numpy as np
from uuid import uuid4
from typing import List, Dict

from sms.src.log import configure_logging
from sms.src.vector_search.evaluate_top_k import build_model, create_embedding_dict
from sms.exp1.config_classes import LaunchPlanConfig, load_config_from_launchplan
from pydantic import BaseModel

logger = logging.getLogger(__name__)
configure_logging()

class ModelEmbeddingConfig(BaseModel):
    name: str
    lp_config: LaunchPlanConfig
    mod_path: str
    path_type: str    # 'full' or 'encoder'
    use_full_model: bool

def create_data_dict(val_data_path: str) -> Dict[str, np.ndarray]:
    data = torch.load(val_data_path)
    data_ids = [str(uuid4()) for _ in range(len(data))]
    return dict(zip(data_ids, data))

def create_and_save_embeddings(
    data_dict: Dict[str, np.ndarray],
    model_config: ModelEmbeddingConfig,
    output_folder: Path,
    batch_size: int = 256
) -> None:
    logger.info(f"Creating embeddings for {model_config.name}")

    dumped_lp_config = model_config.lp_config.model_dump()
    bm_cfg = {'full_model_path': model_config.mod_path} if model_config.path_type == 'full' else {'encoder_path': model_config.mod_path}

    model = build_model(dumped_lp_config, **bm_cfg, use_full_model=model_config.use_full_model)
    embeddings_dict = create_embedding_dict(data_dict, dumped_lp_config, model, batch_size)

    output_file = output_folder / f"{model_config.name}_embeddings.pt"
    torch.save(embeddings_dict, output_file)
    logger.info(f"Saved embeddings for {model_config.name} to {output_file}")

def process_run_folder(run_folder: Path, data_dict: Dict[str, np.ndarray], batch_size: int) -> None:
    logger.info(f"Processing run folder: {run_folder}")

    embeddings_folder = run_folder / "embeddings"
    embeddings_folder.mkdir(exist_ok=True)

    lp_config = load_config_from_launchplan(run_folder / "original_launchplan.yaml")

    model_configs = []
    
    # Pretrain models
    pretrain_models = [
        ("pretrain_saved_model.pth", "pretrain_embeddings.pt"),
        ("pretrain_saved_model_last.pth", "pretrain_embeddings_last.pt")
    ]
    for model_file, _ in pretrain_models:
        if (run_folder / model_file).exists():
            model_configs.append(ModelEmbeddingConfig(
                name=f"{run_folder.name}_{model_file.split('.')[0]}",
                lp_config=lp_config,
                mod_path=str(run_folder / model_file),
                path_type='full',
                use_full_model=True
            ))
    
    # Finetune models
    finetune_models = [
        ("finetune_saved_model.pth", "finetune_embeddings.pt"),
        ("finetune_saved_model_last.pth", "finetune_embeddings_last.pt")
    ]
    for model_file, _ in finetune_models:
        if (run_folder / model_file).exists():
            model_configs.append(ModelEmbeddingConfig(
                name=f"{run_folder.name}_{model_file.split('.')[0]}",
                lp_config=lp_config,
                mod_path=str(run_folder / model_file),
                path_type='encoder',
                use_full_model=False
            ))
    
    for config in model_configs:
        create_and_save_embeddings(data_dict, config, embeddings_folder, batch_size)

def main():
    val_data_path = "data/exp1/val_data.pt"
    runs_dir = Path("sms/exp1/runs")
    batch_size = 32

    # Create data_dict
    data_dict = create_data_dict(val_data_path)
    logger.info(f"Created data_dict with {len(data_dict)} entries")

    # Process each run folder
    for run_folder in runs_dir.iterdir():
        if run_folder.is_dir():
            process_run_folder(run_folder, data_dict, batch_size)

    logger.info("Embedding creation completed for all run folders")

if __name__ == "__main__":
    main()