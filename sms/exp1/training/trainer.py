import os
import torch
import logging
from dotenv import load_dotenv
from typing import Callable

from torch.utils.data import DataLoader
import wandb
import time

import sms.exp1.training.loss_functions as loss_functions
from sms.exp1.config_classes import LaunchPlanConfig
from sms.src.log import configure_logging

load_dotenv()

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
            model_save_path: str,
            run_folder: str = None,
            early_stopping_patience: int = 5,
            mode='pretrain'
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
        self.model_save_path = model_save_path
        self.mode = mode
        self.current_model_path = os.path.join(self.run_folder, f'{self.mode}_saved_model_last.pt')

        if self.mode not in ['pretrain', 'finetune']:
            raise ValueError("Mode must be either 'pretrain' or 'finetune'")

        if self.mode == 'pretrain':
            self.step = self.pretrain_step
        else:
            self.step = self.finetune_step
        
        self.logger = logging.getLogger(__name__)
        configure_logging(console_level=logging.INFO)

        # Initialize Weights & Biases
        # wandb.init(project=os.getenv('WANDB_PROJECT'), config=config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.best_loss = float('inf')
        self.early_stopping_counter = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"{name} requires_grad: {param.requires_grad}")

    def pretrain_step(self, batch):
        """
        Uses VICReg loss with a positive example.
        """
        # print("NEW BATCH \n \n \n \n")
        anchor, positive = batch
        anchor, positive = anchor.float().to(self.device).requires_grad_(), positive.float().to(self.device).requires_grad_()
        # print(f"Anchor requires_grad: {anchor.requires_grad}")
        # print(f"Positive requires_grad: {positive.requires_grad}")
        proj_anchor, proj_positive = self.model(anchor, positive)
        loss, loss_inv, loss_var, loss_cov = self.loss([proj_anchor, proj_positive])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
    def finetune_step(self, batch):
        """
        Uses contrastive loss with a positive and negative example.
        """
        anchor, positive, negative = batch
        anchor, positive, negative = anchor.float().to(self.device).requires_grad_(), positive.float().to(self.device).requires_grad_(), negative.float().to(self.device).requires_grad_()
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

    def train(self):
        metrics = {
            'epochs': [],
            'train_loss': [],
            'time_taken': []  # New metric for time taken
        }
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch()
            self.scheduler.step(train_loss)
            end_time = time.time()
            time_taken = end_time - start_time
            
            metrics['epochs'].append(epoch)
            metrics['train_loss'].append(train_loss)
            metrics['time_taken'].append(time_taken)  # Record time taken

            self.logger.info(
                f'Epoch {epoch}/{self.epochs} | Train Loss: {train_loss:.4f} | Time: {time_taken:.2f}s'
            )

            # Log metrics to wandb
            # if wandb.run is not None:
            #     wandb.log({
            #         'epoch': epoch,
            #         'train_loss': train_loss,
            #         'time_taken': time_taken  # Log time taken to wandb
            #     })
            
            # Save the current model after each epoch
        
            if self.mode == 'pretrain':
                torch.save(self.model.state_dict(), self.current_model_path)
            else:
                torch.save(self.model.get_encoder().state_dict(), self.current_model_path)
            self.logger.info(f'Saved current model for epoch {epoch}.')
        
            # Save the best model
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                if self.mode == 'pretrain':
                    torch.save(self.model.state_dict(), self.model_save_path)
                else:
                    torch.save(self.model.get_encoder().state_dict(), self.model_save_path)
                self.logger.info('Best model saved.')
            else:
                self.early_stopping_counter += 1
                self.logger.info(f'No improvement. Early Stopping Counter: {self.early_stopping_counter}')
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info('Early stopping triggered.')
                    break

        # if wandb.run is not None:
        #     wandb.finish()

        return metrics