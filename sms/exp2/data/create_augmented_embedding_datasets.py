import logging
import yaml
import os
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict, Any, Tuple

from sms.src.log import configure_logging
from sms.src.vector_search.evaluate_top_k import create_augmented_data, create_embedding_dict, build_model
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

def select_keys(keys: List[str], all_keys: List[str], shuffle: bool = True) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Shuffle the keys and select the required number of keys for each dataset size.
    
    Returns two dictionaries:
    - subset_keys: Maps dataset sizes to their subset of keys.
    - selected_keys: Maps dataset sizes to their selected keys for augmentation.
    """
    if shuffle:
        np.random.shuffle(keys)
        logger.info("Shuffled the dataset keys.")
    
    selection_plan = [
        (5000, 100, '5k'),
        (10000, 200, '10k'),
        (100000, 1000, '100k'),
        (500000, 2000, '500k'),
        (None, 10000, '1m') 
    ]

    subset_keys = {}
    selected_keys = {}
    start = 0
    total_keys = len(keys)

    for size, count, label in selection_plan:
        if size is not None:
            end = start + size
            subset = keys[start:end]
            subset_keys[label] = subset
            selected = subset[:count]
            selected_keys[label] = selected
            logger.info(f"Selected {len(selected)} keys for dataset size {label} from subset of {len(subset)} keys.")
            start = end
            if start >= total_keys:
                logger.warning(f"Reached the end of the dataset at size {label}.")
                break
        else:
            # For 1m, select from the remaining keys
            selected = keys[start:start + count]
            selected_keys[label] = selected
            subset_keys[label] = all_keys
            logger.info(f"Selected {len(selected)} keys for dataset size {label} from the entire remaining dataset.")
    
    return subset_keys, selected_keys

def precompute_augmentations(data_dict: Dict[str, np.ndarray], selected_keys: Dict[str, List[str]]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Precompute augmentations for the selected keys.
    
    Returns a dictionary mapping dataset sizes to their augmented data dictionaries. These dictionaries map data ids to dictionaries of augmented data.
    """
    precomputed_augmentations = {}
    
    for size, keys in selected_keys.items():
        logger.info(f"Creating augmentations for dataset size {size} with {len(keys)} anchors.")
        augmented_data = create_augmented_data(data_dict, keys)
        precomputed_augmentations[size] = augmented_data
        logger.info(f"Created augmentations for dataset size {size}.")
    
    return precomputed_augmentations

def precompute_embeddings(
    augmented_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    model_configs: List[ModelEmbeddingConfig],
    output_folder: Path,
    batch_size: int = 256
) -> None:
    """
    Create embeddings for the augmented data using the specified models and save them.
    """
    for size, nested_aug_dict in augmented_data.items():
        logger.info(f"Processing embeddings for dataset size {size}.")
        for config in model_configs:
            logger.info(f"Creating embeddings for model: {config.name} on dataset size {size}")
            dumped_lp_config = config.lp_config.model_dump()
            bm_cfg = {'full_model_path': config.mod_path} if config.path_type == 'full' else {'encoder_path': config.mod_path}

            model = build_model(dumped_lp_config, **bm_cfg, use_full_model=config.use_full_model)
            
            augmented_embeddings_dict = {}
            for data_id, aug_dict in nested_aug_dict.items():
                print(f"Type of aug_dict: {type(aug_dict)}")
                augmented_embeddings_dict[data_id] = create_embedding_dict(aug_dict, dumped_lp_config, model, batch_size=batch_size)
            
            # Define the output path
            augmented_size_str = size
            output_file = output_folder / f"{config.name}_aug_{augmented_size_str}_embeddings.pt"
            torch.save(augmented_embeddings_dict, output_file)
            logger.info(f"Saved embeddings to {output_file}")

def save_to_disk(data: Any, file_path: Path) -> None:
    """
    Save the given data to a PyTorch file if it doesn't already exist.
    """
    if file_path.exists():
        logger.info(f"File {file_path} already exists. Skipping save.")
        return

    try:
        torch.save(data, file_path)
        logger.info(f"Saved data to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")

def main():
    data_path = Path("data/exp2/million_chunks.pt")
    output_dir = Path("data/exp2/augmented_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    try:
        data_dict = torch.load(data_path)
        logger.info(f"Loaded data with {len(data_dict)} entries.")
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return
    
    # make sure the augmentations have at least 3 notes
    keys = [k for k, v in data_dict.items() if len(v) >= 3]
    all_keys = list(data_dict.keys())

    subset_keys_path = output_dir / "subset_keys.pt"
    selected_keys_path = output_dir / "selected_keys.pt"

    if subset_keys_path.exists() and selected_keys_path.exists():
        logger.info("Loading existing subset_keys and selected_keys")
        subset_keys = torch.load(subset_keys_path)
        selected_keys = torch.load(selected_keys_path)
    else:
        # Select keys for augmentations
        subset_keys, selected_keys = select_keys(keys, all_keys)

        # Save the subset keys
        save_to_disk(subset_keys, subset_keys_path)

        # Save the selected keys
        save_to_disk(selected_keys, selected_keys_path)

    # Check if precomputed_augmentations already exist
    augmentations_path = output_dir / "precomputed_augmentations.pt"

    if augmentations_path.exists():
        logger.info("Loading existing precomputed_augmentations")
        precomputed_augmentations = torch.load(augmentations_path)
    else:
        # Precompute augmentations
        precomputed_augmentations = precompute_augmentations(data_dict, selected_keys)

        # Save the augmentations
        save_to_disk(precomputed_augmentations, augmentations_path)

    logger.info("Subset keys, selected keys, and precomputed augmentations are ready.")

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

    # Precompute embeddings
    precompute_embeddings(precomputed_augmentations, model_configs, output_dir)

    logger.info("Precomputed embeddings for all dataset sizes and models.")

if __name__ == "__main__":
    main()