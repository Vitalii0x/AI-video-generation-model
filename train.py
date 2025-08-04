#!/usr/bin/env python3
"""
Main training script for text-to-video generation model.
"""

import argparse
import torch
import torch.nn as nn
import os
import json
from omegaconf import OmegaConf

from models import TextToVideoModel
from data import create_synthetic_dataloader
from training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train text-to-video generation model')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of video frames')
    parser.add_argument('--height', type=int, default=256, help='Video height')
    parser.add_argument('--width', type=int, default=256, help='Video width')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Data arguments
    parser.add_argument('--synthetic_samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--use_real_data', action='store_true', help='Use real video data')
    parser.add_argument('--data_dir', type=str, default='', help='Directory containing video data')
    parser.add_argument('--annotation_file', type=str, default='', help='Annotation file path')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='Checkpoint save interval')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--config', type=str, default='', help='Configuration file')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup training environment."""
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def create_model(args):
    """Create the text-to-video model."""
    model = TextToVideoModel(
        d_model=args.d_model,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_timesteps=args.num_timesteps,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    return model


def create_dataloaders(args):
    """Create training and validation dataloaders."""
    if args.use_real_data and args.data_dir and args.annotation_file:
        # Use real data
        from data import create_dataloader
        train_loader, val_loader, test_loader = create_dataloader(
            data_dir=args.data_dir,
            annotation_file=args.annotation_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            split_ratio=(0.8, 0.1, 0.1)
        )
    else:
        # Use synthetic data
        train_loader, val_loader, test_loader = create_synthetic_dataloader(
            num_samples=args.synthetic_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            split_ratio=(0.8, 0.1, 0.1)
        )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration file if provided
    if args.config:
        config = OmegaConf.load(args.config)
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Setup environment
    setup_environment(args)
    
    # Create model
    model = create_model(args)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(args)
    print(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create trainer
    trainer_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'output_dir': args.output_dir,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from {args.resume}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main() 