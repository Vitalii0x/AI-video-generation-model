import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional

from models import TextToVideoModel


class Trainer:
    """Main trainer class for text-to-video generation model."""
    
    def __init__(self, 
                 model: TextToVideoModel,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Dict[str, Any] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup training components
        self._setup_training()
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=len(self.train_loader) * self.config.get('num_epochs', 100)
        )
        
    def _setup_logging(self):
        """Setup logging and experiment tracking."""
        # Create output directory
        self.output_dir = self.config.get('output_dir', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            videos = batch['video'].to(self.device)
            texts = batch['text']
            
            # Sample random timesteps
            batch_size = videos.shape[0]
            t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Add noise to videos
            noise = torch.randn_like(videos)
            noisy_videos = self.model.diffusion.q_sample(videos, t, noise)
            
            # Predict noise
            predicted_noise = self.model(noisy_videos, texts, t)
            
            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to TensorBoard
            if self.global_step % self.config.get('log_interval', 100) == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]["lr"], self.global_step)
                
        self.scheduler.step()
        return epoch_loss / num_batches
        
    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None
            
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                videos = batch['video'].to(self.device)
                texts = batch['text']
                
                # Sample random timesteps
                batch_size = videos.shape[0]
                t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
                
                # Add noise to videos
                noise = torch.randn_like(videos)
                noisy_videos = self.model.diffusion.q_sample(videos, t, noise)
                
                # Predict noise
                predicted_noise = self.model(noisy_videos, texts, t)
                
                # Compute loss
                loss = F.mse_loss(predicted_noise, noise)
                val_loss += loss.item()
                
        val_loss /= num_batches
        
        # Log validation metrics
        self.writer.add_scalar('val/loss', val_loss, self.global_step)
            
        return val_loss
        
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            
    def train(self):
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 100)
        
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Log epoch metrics
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f if val_loss else 'N/A'}")
            
            # Save checkpoint
            is_best = val_loss and val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            self.save_checkpoint(is_best)
                    
        print("Training completed!")
        
        # Close logging
        self.writer.close() 