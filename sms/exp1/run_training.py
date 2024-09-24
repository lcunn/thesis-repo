import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from sms.exp1.training.trainer import Trainer
from sms.exp1.data.dataloader import OneBarChunkDataset, get_dataloader
from sms.exp1.models.encoders_conv import ConvPianoRollConfig, ConvQuantizedTimeConfig
from sms.exp1.models.siamese import SiameseModel

"""
config has the following structure:

input: {
    format: {
        normalize_octave: bool
        make_relative_pitch: bool
        quantize: bool
        piano_roll: bool
        steps_per_bar: int
        rest_pitch: int
    }
    input_size: 
}
encoder: {
    type: str
    layers: List[Dict]
}
projector: List[Dict] 
training: {
    pretraining_loss: {
        type: str
        params: Dict
    }
    finetuning_loss: {
        type: str
        params: Dict
    }
}
optimizer: {}
scheduler: {}
metrics: {}
device: str
num_epochs: int
batch_size: int
num_workers: int
data_paths: [str]
hp: {
    d_latent: int
    d_projected: int
}




"""

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path: str):
    # Load configuration
    config = load_config(config_path)

    train_loader = get_dataloader(
        data_paths=config['data_paths'], 
        format_config=config['format_config'], 
        use_transposition=config['use_transposition'], 
        neg_enhance=config['neg_enhance'], 
        split='train',
        split_ratio=config['split_ratio'],
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'],
        use_sequence_collate_fn=False,
        shuffle=True
        )
    
    val_loader = get_dataloader(
        data_paths=config['data_paths'], 
        format_config=config['format_config'], 
        use_transposition=config['use_transposition'], 
        neg_enhance=config['neg_enhance'], 
        split='val',
        split_ratio=config['split_ratio'],
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'],
        use_sequence_collate_fn=False,
        shuffle=True
        )

    # Initialize Model
    model = SiameseModel(
        input_dim=config['model']['input_dim'],
        embedding_dim=config['model']['embedding_dim'],
        architecture=config['model']['architecture']
    )

    # Initialize Trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Start Training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Training for Siamese Network with VICReg Loss")
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Path to the training configuration file.'
    )
    args = parser.parse_args()
    main(args.config)