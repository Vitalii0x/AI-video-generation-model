#!/usr/bin/env python3
"""
Example script demonstrating text-to-video generation.
"""

import torch
import numpy as np
import os
from models import TextToVideoModel
from data import create_synthetic_dataloader
from utils.video_utils import save_video, create_video_grid


def main():
    """Main example function."""
    print("Text-to-Video Generation Example")
    print("=" * 40)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\n1. Creating model...")
    model = TextToVideoModel(
        d_model=768,
        num_frames=16,
        height=256,
        width=256,
        num_timesteps=100,  # Reduced for faster generation
        num_heads=8,
        num_layers=6  # Reduced for faster generation
    )
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create synthetic dataloader for testing
    print("\n2. Creating synthetic dataset...")
    train_loader, val_loader, test_loader = create_synthetic_dataloader(
        num_samples=100,
        batch_size=4,
        num_frames=16,
        height=256,
        width=256,
        split_ratio=(0.8, 0.1, 0.1)
    )
    print(f"Created dataloaders: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    model.train()
    batch = next(iter(train_loader))
    videos = batch['video'].to(device)
    texts = batch['text']
    
    # Sample random timesteps
    batch_size = videos.shape[0]
    t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
    
    # Forward pass
    with torch.no_grad():
        # Add noise to videos
        noise = torch.randn_like(videos)
        noisy_videos = model.diffusion.q_sample(videos, t, noise)
        
        # Predict noise
        predicted_noise = model(noisy_videos, texts, t)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
    print(f"Forward pass successful! Loss: {loss.item():.4f}")
    
    # Test video generation
    print("\n4. Testing video generation...")
    model.eval()
    
    # Example texts
    example_texts = [
        "A cat playing with a ball",
        "A dog running in the park",
        "A bird flying in the sky",
        "A car driving on the road"
    ]
    
    with torch.no_grad():
        # Generate videos (this will be random since model is not trained)
        generated_videos = model.sample(
            example_texts,
            batch_size=len(example_texts),
            num_frames=16
        )
        
        # Convert to numpy
        videos_np = generated_videos.cpu().numpy()
        videos_np = (videos_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        videos_np = np.clip(videos_np, 0, 1)
        
        # Convert to [F, H, W, C] format for saving
        videos_np = np.transpose(videos_np, (0, 2, 3, 4, 1))  # [B, C, F, H, W] -> [B, F, H, W, C]
    
    # Save individual videos
    print("\n5. Saving generated videos...")
    os.makedirs('example_outputs', exist_ok=True)
    
    for i, (video, text) in enumerate(zip(videos_np, example_texts)):
        output_path = f'example_outputs/generated_video_{i}.mp4'
        save_video(video, output_path, fps=8)
        print(f"Saved video {i+1}: '{text}' -> {output_path}")
    
    # Create video grid
    print("\n6. Creating video grid...")
    grid_path = 'example_outputs/video_grid.mp4'
    create_video_grid(videos_np, grid_shape=(2, 2), output_path=grid_path, fps=8)
    print(f"Created video grid: {grid_path}")
    
    print("\nExample completed successfully!")
    print("Check the 'example_outputs' directory for generated videos.")


if __name__ == '__main__':
    main() 