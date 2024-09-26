from typing import Dict
from pathlib import Path
import yaml
from pydantic import BaseModel

class InputConfig(BaseModel):
    normalize_octave: bool = False
    make_relative_pitch: bool = False
    quantize: bool = False
    piano_roll: bool = False
    steps_per_bar: int = 32
    rest_pitch: int = -1

class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    train_data_path: str
    val_data_path: str
    use_transposition: bool = False
    neg_enhance: bool = True
    use_sequence_collate_fn: bool = False
    shuffle: bool = True

class EncoderConfig(BaseModel):
    type: str
    params: Dict

class ProjectorConfig(BaseModel):
    params: Dict

class LossConfig(BaseModel):
    type: str
    params: Dict

class OptimizerConfig(BaseModel):
    type: str
    params: Dict

class SchedulerConfig(BaseModel):
    type: str
    params: Dict

class DimensionsConfig(BaseModel):
    input_shape: tuple
    d_latent: int
    d_projected: int

class TrainingConfig(BaseModel):
    epochs: int
    early_stopping_patience: int

class LaunchPlanConfig(BaseModel):
    input: InputConfig
    pt_dl: DataLoaderConfig
    ft_dl: DataLoaderConfig
    encoder: EncoderConfig
    projector: ProjectorConfig
    pt_loss: LossConfig
    ft_loss: LossConfig
    dims: DimensionsConfig
    pt_optimizer: OptimizerConfig
    ft_optimizer: OptimizerConfig
    pt_scheduler: SchedulerConfig
    ft_scheduler: SchedulerConfig
    pt_training: TrainingConfig
    ft_training: TrainingConfig

def load_config_from_launchplan(lp_yaml_path: str) -> LaunchPlanConfig:
    lp_yaml_path = Path(lp_yaml_path)

    with open(lp_yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # load nested YAML files
    for key, value in config_dict.items():
        if isinstance(value, str) and value.endswith('.yaml'):
            with open(value, 'r') as f:
                config_dict[key] = yaml.safe_load(f)

    return LaunchPlanConfig.model_validate(config_dict)