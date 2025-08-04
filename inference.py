#!/usr/bin/env python3
"""
Inference script for text-to-video generation.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from typing import List, Optional
import json

from models import TextToVideoModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate videos from text')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of video frames')
    parser.add_argument('--height', type=int, default=256, help='Video height')
    parser.add_argument('--width', type=int, default=256, help='Video width')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    
    # Generation arguments
    parser.add_argument('--text', type=str, required=True, help='Text description for video generation')
    parser.add_argument('--output_path', type=str, default='generated_video.mp4', help='Output video path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for generation')
    parser.add_argument('--fps', type=int, default=8, help='Output video FPS')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    
    return parser.parse_args()


def setup_device(device_arg: str):
    """Setup device for inference."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def load_model(checkpoint_path: str, args, device: torch.device):
    """Load trained model from checkpoint."""
    # Create model
    model = TextToVideoModel(
        d_model=args.d_model,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_timesteps=args.num_timesteps,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    return model


def generate_video(model: TextToVideoModel, text: str, args, device: torch.device):
    """Generate video from text."""
    print(f"Generating video for text: '{text}'")
    
    with torch.no_grad():
        # Generate video
        video = model.sample(
            text=[text] * args.batch_size,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width
        )
        
        # Take first video if batch_size > 1
        if args.batch_size > 1:
            video = video[0]
        
        # Convert to numpy and denormalize
        video_np = video.cpu().numpy()
        video_np = (video_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        video_np = np.clip(video_np, 0, 1)
        
        return video_np


def save_video(video: np.ndarray, output_path: str, fps: int = 8):
    """Save video array to file."""
    # Video parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Get video dimensions
    channels, frames, height, width = video.shape
    
    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame_idx in range(frames):
        frame = video[:, frame_idx, :, :]  # [C, H, W]
        frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
    out.release()
    print(f"Saved video to {output_path}")


def save_frames(video: np.ndarray, output_dir: str):
    """Save individual frames as images."""
    os.makedirs(output_dir, exist_ok=True)
    
    channels, frames, height, width = video.shape
    
    for frame_idx in range(frames):
        frame = video[:, frame_idx, :, :]  # [C, H, W]
        frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
        frame = (frame * 255).astype(np.uint8)
        
        # Save as PNG
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    print(f"Saved {frames} frames to {output_dir}")


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load model
    model = load_model(args.checkpoint, args, device)
    
    # Generate video
    video = generate_video(model, args.text, args, device)
    
    # Save video
    save_video(video, args.output_path, args.fps)
    
    # Save individual frames (optional)
    frames_dir = args.output_path.replace('.mp4', '_frames')
    save_frames(video, frames_dir)
    
    print("Video generation completed!")


if __name__ == '__main__':
    main() 