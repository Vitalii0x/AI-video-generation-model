import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from PIL import Image
import json
from typing import List, Dict, Tuple, Optional
import random


class VideoDataset(Dataset):
    """Dataset for video-text pairs."""
    
    def __init__(self, 
                 data_dir: str,
                 annotation_file: str,
                 num_frames: int = 16,
                 height: int = 256,
                 width: int = 256,
                 frame_rate: int = 8,
                 max_text_length: int = 77,
                 transform=None):
        """
        Args:
            data_dir: Directory containing video files
            annotation_file: JSON file with video-text annotations
            num_frames: Number of frames to sample from each video
            height: Target video height
            width: Target video width
            frame_rate: Target frame rate
            max_text_length: Maximum text sequence length
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        self.max_text_length = max_text_length
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # Filter valid videos
        self.valid_samples = []
        for item in self.annotations:
            video_path = os.path.join(data_dir, item['video_path'])
            if os.path.exists(video_path):
                self.valid_samples.append(item)
                
        print(f"Loaded {len(self.valid_samples)} valid video-text pairs")
        
    def __len__(self):
        return len(self.valid_samples)
        
    def __getitem__(self, idx):
        """Get a video-text pair."""
        item = self.valid_samples[idx]
        
        # Load video
        video_path = os.path.join(self.data_dir, item['video_path'])
        video_frames = self._load_video(video_path)
        
        # Load text
        text = item['text']
        
        # Apply transforms
        if self.transform:
            video_frames = self.transform(video_frames)
            
        return {
            'video': video_frames,
            'text': text,
            'video_path': video_path
        }
        
    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video frames."""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, (self.width, self.height))
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
            frame_count += 1
            
        cap.release()
        
        if len(frames) == 0:
            # Return dummy video if loading fails
            return torch.zeros(self.num_frames, self.height, self.width, 3)
            
        # Sample frames uniformly
        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Pad with last frame if video is too short
            last_frame = frames[-1]
            while len(frames) < self.num_frames:
                frames.append(last_frame)
                
        # Convert to tensor: [num_frames, height, width, channels]
        video_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
        
        # Transpose to [channels, num_frames, height, width]
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        
        return video_tensor


class SyntheticVideoDataset(Dataset):
    """Synthetic dataset for testing with random videos and text."""
    
    def __init__(self, 
                 num_samples: int = 1000,
                 num_frames: int = 16,
                 height: int = 256,
                 width: int = 256,
                 text_templates: Optional[List[str]] = None):
        """
        Args:
            num_samples: Number of synthetic samples to generate
            num_frames: Number of frames per video
            height: Video height
            width: Video width
            text_templates: List of text templates for generation
        """
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        
        # Default text templates
        if text_templates is None:
            self.text_templates = [
                "A cat playing with a ball",
                "A dog running in the park",
                "A bird flying in the sky",
                "A car driving on the road",
                "A person walking down the street",
                "A flower blooming in the garden",
                "A tree swaying in the wind",
                "A river flowing through the valley",
                "A mountain peak covered in snow",
                "A sunset over the ocean"
            ]
        else:
            self.text_templates = text_templates
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        """Generate a synthetic video-text pair."""
        # Generate random video
        video = torch.randn(3, self.num_frames, self.height, self.width)
        
        # Select random text
        text = random.choice(self.text_templates)
        
        return {
            'video': video,
            'text': text,
            'video_path': f'synthetic_{idx}.mp4'
        }


class VideoTextDataset(Dataset):
    """Combined dataset that can load from multiple sources."""
    
    def __init__(self, 
                 datasets: List[Dataset],
                 weights: Optional[List[float]] = None):
        """
        Args:
            datasets: List of datasets to combine
            weights: Sampling weights for each dataset
        """
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        
        # Calculate cumulative lengths
        self.cumulative_lengths = []
        total_length = 0
        for dataset in datasets:
            total_length += len(dataset)
            self.cumulative_lengths.append(total_length)
            
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
        
    def __getitem__(self, idx):
        """Get item from weighted random dataset."""
        # Select dataset based on weights
        dataset_idx = random.choices(
            range(len(self.datasets)), 
            weights=self.weights, 
            k=1
        )[0]
        
        # Get item from selected dataset
        dataset = self.datasets[dataset_idx]
        item_idx = idx % len(dataset)
        
        return dataset[item_idx] 