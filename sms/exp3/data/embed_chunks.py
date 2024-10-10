import logging
import yaml
import os
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict, Any

from sms.src.log import configure_logging
from sms.src.vector_search.evaluate_top_k import build_model, create_embedding_dict
from sms.exp1.config_classes import LaunchPlanConfig, load_config_from_launchplan
from pydantic import BaseModel

logger = logging.getLogger(__name__)
configure_logging()

def create_and_save_embeddings(
    data_dict: Dict[str, Dict[Any, np.ndarray]],
    model: str,
    dumped_lp_config: Dict[str, Any],
    output_path: str,
    batch_size: int = 512
) -> None:
    
    output_dict = {}
    for i, (song_id, scale_dict) in enumerate(data_dict.items()):
        logger.info(f"Creating embedding {i}/{len(data_dict)}")
        song_dict = {}
        for scale, chunk_list in scale_dict.items():
            temp_chunk_dict = {}
            for chunk_idx, chunk in enumerate(chunk_list):
                temp_chunk_dict[chunk_idx] = chunk
            embeddings_dict = create_embedding_dict(temp_chunk_dict, dumped_lp_config, model, batch_size)
            song_dict[scale] = list(embeddings_dict.values())
        output_dict[song_id] = song_dict
        
    torch.save(output_dict, output_path)
    logger.info(f"Saved embeddings to {output_path}")

def main():
    # data_path = Path("data/exp3/maestro_chunks.pt")
    # data_dict = torch.load(data_path)

    # lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_rel_1/original_launchplan.yaml")
    # mod_path="sms/exp1/runs/transformer_rel_1/pretrain_saved_model.pth"

    # model = build_model(lp_config.model_dump(), full_model_path=mod_path, use_full_model=True)

    # create_and_save_embeddings(data_dict, model, lp_config.model_dump(), "data/exp3/embeddings/transformer_rel_1.pt")

    # data_path = Path("data/exp3/maestro_chunks.pt")
    # data_dict = torch.load(data_path)

    # lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_quant_rel_bigenc_1/original_launchplan.yaml"),
    # mod_path="sms/exp1/runs/transformer_quant_rel_bigenc_1/pretrain_saved_model.pth",

    # model = build_model(lp_config.model_dump(), full_model_path=mod_path, use_full_model=True)

    # create_and_save_embeddings(data_dict, model, lp_config.model_dump(), "data/exp3/embeddings/transformer_quant_rel_bigenc_best_1.pt")

    # lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_quant_rel_bigenc_1/original_launchplan.yaml"),
    # mod_path="sms/exp1/runs/transformer_quant_rel_bigenc_1/pretrain_saved_model_last.pt",

    # model = build_model(lp_config.model_dump(), full_model_path=mod_path, use_full_model=True)

    # create_and_save_embeddings(data_dict, model, lp_config.model_dump(), "data/exp3/embeddings/transformer_quant_rel_bigenc_last_1.pt")

    data_path = Path("data/exp3/all_plag_chunks.pt")
    data_dict = torch.load(data_path)

    lp_config=load_config_from_launchplan("sms/exp1/runs/transformer_quant_rel_bigenc_1/original_launchplan.yaml")
    mod_path="sms/exp1/runs/transformer_quant_rel_bigenc_1/pretrain_saved_model.pth"

    model = build_model(lp_config.model_dump(), full_model_path=mod_path, use_full_model=True)

    create_and_save_embeddings(data_dict, model, lp_config.model_dump(), "data/exp3/all_plag_chunks_embeddings_qrb_best.pt")

if __name__ == "__main__":
    main()
