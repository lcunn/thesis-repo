import logging
import yaml
import os
from pathlib import Path
import torch
import numpy as np
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

def create_and_save_embeddings(
    data_dict: Dict[str, np.ndarray],
    model_config: ModelEmbeddingConfig,
    output_folder: Path,
    batch_size: int = 256,
    max_embeddings_per_file: int = 500000
) -> None:
    logger.info(f"Creating embeddings for {model_config.name}")

    dumped_lp_config = model_config.lp_config.model_dump()
    bm_cfg = {'full_model_path': model_config.mod_path} if model_config.path_type == 'full' else {'encoder_path': model_config.mod_path}

    model = build_model(dumped_lp_config, **bm_cfg, use_full_model=model_config.use_full_model)
    
    # Process data in batches and save to multiple files
    data_items = list(data_dict.items())
    for i in range(0, len(data_items), max_embeddings_per_file):
        batch_data = dict(data_items[i:i+max_embeddings_per_file])
        embeddings_dict = create_embedding_dict(batch_data, dumped_lp_config, model, batch_size)
        logger.info(f"Created embeddings for {len(embeddings_dict)} keys")
        
        output_file = output_folder / f"{model_config.name}_embeddings_{i//max_embeddings_per_file}.pt"
        torch.save(embeddings_dict, output_file)
        logger.info(f"Saved embeddings batch to {output_file}")

def main():
    data_path = Path("data/exp2/million_chunks.pt")
    output_dir = Path("data/exp2/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    data_dict = torch.load(data_path)
    logger.info(f"Loaded data with {len(data_dict)} entries")

    # Define model configurations
    # model_configs = [
    #     ModelEmbeddingConfig(
    #         name="transformer_rel_1_pretrain",
    #         lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_rel_1/original_launchplan.yaml"),
    #         mod_path="sms/exp1/runs/transformer_rel_1/pretrain_saved_model.pth",
    #         path_type='full',
    #         use_full_model=True
    #     ),
    #     ModelEmbeddingConfig(
    #         name="transformer_pr_1_pretrain",
    #         lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_pr_1/original_launchplan.yaml"),
    #         mod_path="sms/exp1/runs/transformer_pr_1/pretrain_saved_model.pth",
    #         path_type='full',
    #         use_full_model=True
    #     )
    # ]

    model_configs = [
        ModelEmbeddingConfig(
            name="transformer_quant_rel_bigenc_1_pretrain_best",
            lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_quant_rel_bigenc_1/original_launchplan.yaml"),
            mod_path="sms/exp1/runs/transformer_quant_rel_bigenc_1/pretrain_saved_model.pth",
            path_type='full',
            use_full_model=True
        ),
        ModelEmbeddingConfig(
            name="transformer_quant_rel_bigenc_1_finetune_last",
            lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_quant_rel_bigenc_1/original_launchplan.yaml"),
            mod_path="sms/exp1/runs/transformer_quant_rel_bigenc_1/pretrain_saved_model_last.pt",
            path_type='full',
            use_full_model=True
        )
    ]

    for config in model_configs:
        create_and_save_embeddings(data_dict, config, output_dir)

    logger.info("Embedding creation completed for all models")

if __name__ == "__main__":
    main()