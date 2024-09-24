import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from wandb_utils import WandbCallback  # Assumes you have a wandb_utils module

from training.trainer import Trainer
from sms.exp1.data.preprocess import MelodyDataset, collate_fn
from sms.exp1.models.siamese_model import SiameseModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Initialize Datasets and DataLoaders
    train_dataset = MelodyDataset(
        data_dir=config['data_dir'],
        split='train',
        transform=config['transforms']
    )
    val_dataset = MelodyDataset(
        data_dir=config['data_dir'],
        split='val',
        transform=config['transforms']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
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