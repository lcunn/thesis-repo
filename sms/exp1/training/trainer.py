import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import wandb

from sms.exp1.training.loss_functions import vicreg_loss
from sms.src.log import configure_logging
from sms.exp1.models.siamese import SiameseModel

class Trainer:
    def __init__(self, config, model: SiameseModel, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        configure_logging()

        # Initialize Weights & Biases
        wandb.init(project=config['wandb']['project'], config=config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.criterion = vicreg_loss(
            weight_inv=config['loss']['weight_inv'],
            weight_var=config['loss']['weight_var'],
            weight_cov=config['loss']['weight_cov']
        )
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['optimizer']['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            verbose=True
        )

        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            anchor, positive = batch['anchors'].to(self.device), batch['positives'].to(self.device)

            self.optimizer.zero_grad()
            embed_anchor, embed_positive = self.model(anchor, positive)

            loss = self.criterion(embed_anchor, embed_positive)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                anchor, positive = batch['anchors'].to(self.device), batch['positives'].to(self.device)

                embed_anchor, embed_positive = self.model(anchor, positive)
                loss = self.criterion(embed_anchor, embed_positive)
                running_loss += loss.item()
        
        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        for epoch in range(1, self.config['training']['epochs'] + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            self.scheduler.step(val_loss)

            self.logger.info(
                f'Epoch {epoch}/{self.config["training"]["epochs"]} | '
                f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}'
            )

            # Log metrics to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                self.logger.info('Best model saved.')
                wandb.save('best_model.pth')
            else:
                self.early_stopping_counter += 1
                self.logger.info(f'No improvement. Early Stopping Counter: {self.early_stopping_counter}')
                if self.early_stopping_counter >= self.config['training']['early_stopping_patience']:
                    self.logger.info('Early stopping triggered.')
                    break

        wandb.finish()