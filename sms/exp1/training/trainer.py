import os
import torch
import logging
from typing import Callable

from torch.utils.data import DataLoader
import wandb

import sms.exp1.training.loss_functions as loss_functions
from sms.exp1.config_classes import LaunchPlanConfig
from sms.src.log import configure_logging

class Trainer:
    def __init__(
            self,
            config: LaunchPlanConfig,
            loss: Callable,
            optimizer: torch.optim.Optimizer,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
            scheduler: torch.optim.lr_scheduler,
            early_stopping_patience: int = 5,
            mode='pretrain',
            run_folder: str = None,
            model_save_path: str = None
        ):
        self.epochs = epochs
        self.loss = loss
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.run_folder = run_folder
        self.model_path = model_save_path if model_save_path else os.path.join(self.run_folder, f'best_model_{mode}.pth')

        if mode not in ['pretrain', 'finetune']:
            raise ValueError("Mode must be either 'pretrain' or 'finetune'")

        if mode == 'pretrain':
            self.step = self.pretrain_step
        else:
            self.step = self.finetune_step
        
        self.logger = logging.getLogger(mode)
        configure_logging()

        # Initialize Weights & Biases
        wandb.init(project=config['wandb']['project'], config=config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
    
    def pretrain_step(self, batch):
        """
        Uses VICReg loss with a positive example.
        """
        anchor, positive = batch['anchors'].to(self.device), batch['positives'].to(self.device)
        embed_anchor, embed_positive = self.model(anchor, positive)
        loss = self.loss(embed_anchor, embed_positive)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def finetune_step(self, batch):
        """
        Uses contrastive loss with a positive and negative example.
        """
        anchor, positive, negative = batch['anchors'].to(self.device), batch['positives'].to(self.device), batch['negatives'].to(self.device)
        embed_anchor, embed_positive, embed_negative = self.model(anchor, positive, negative)
        loss = self.loss(embed_anchor, embed_positive, embed_negative)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            loss = self.step(batch)
            running_loss += loss
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.step(batch)
                running_loss += loss
        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': []
        }
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            self.scheduler.step(val_loss)

            metrics['epochs'].append(epoch)
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)

            self.logger.info(
                f'Epoch {epoch}/{self.epochs} | '
                f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}'
            )

            # Log metrics to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                self.logger.info('Best model saved.')
            else:
                self.early_stopping_counter += 1
                self.logger.info(f'No improvement. Early Stopping Counter: {self.early_stopping_counter}')
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info('Early stopping triggered.')
                    break

        if wandb.run is not None:
            wandb.finish()

        return metrics