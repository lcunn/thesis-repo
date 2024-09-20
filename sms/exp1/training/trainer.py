import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..models.embedding_model import EmbeddingModel
from ..models.soft_dtw import SoftDTW
from .loss_functions import CompositeLoss
from ..utils.logger import get_logger

class Trainer:
    def __init__(self, config, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = get_logger('Trainer')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EmbeddingModel(embedding_dim=config.embedding_dim).to(self.device)
        self.soft_dtw = SoftDTW(gamma=config.soft_dtw_gamma).to(self.device)
        self.criterion = CompositeLoss(self.soft_dtw, alpha=config.alpha, beta=config.beta)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            anchor, positive, negative = batch  # Assuming dataset returns triplets
            anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

            self.optimizer.zero_grad()
            embed_anchor = self.model(anchor)
            embed_positive = self.model(positive)
            embed_negative = self.model(negative)

            loss = self.criterion(embed_anchor, embed_positive, embed_negative)
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
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

                embed_anchor = self.model(anchor)
                embed_positive = self.model(positive)
                embed_negative = self.model(negative)

                loss = self.criterion(embed_anchor, embed_positive, embed_negative)
                running_loss += loss.item()
        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            self.scheduler.step(val_loss)

            self.logger.info(f'Epoch {epoch}/{self.config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                self.logger.info('Best model saved.')
            else:
                self.early_stopping_counter += 1
                self.logger.info(f'No improvement. Early Stopping Counter: {self.early_stopping_counter}')
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info('Early stopping triggered.')
                    break