import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional, Tuple
import os

from .dataset import VideoDataset, SyntheticVideoDataset, VideoTextDataset
from .transforms import get_video_transforms


def create_dataloader(data_dir: str,
                     annotation_file: str,
                     batch_size: int = 8,
                     num_workers: int = 4,
                     shuffle: bool = True,
                     num_frames: int = 16,
                     height: int = 256,
                     width: int = 256,
                     augment: bool = True,
                     split_ratio: Optional[Tuple[float, float, float]] = None):
    """
    Create dataloader for video-text dataset.
    
    Args:
        data_dir: Directory containing video files
        annotation_file: JSON file with video-text annotations
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        num_frames: Number of frames per video
        height: Video height
        width: Video width
        augment: Whether to apply data augmentation
        split_ratio: Train/val/test split ratios (e.g., (0.8, 0.1, 0.1))
    Returns:
        DataLoader or tuple of DataLoaders if split_ratio is provided
    """
    # Create transforms
    transforms = get_video_transforms(
        height=height, 
        width=width, 
        augment=augment
    )
    
    # Create dataset
    dataset = VideoDataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        num_frames=num_frames,
        height=height,
        width=width,
        transform=transforms
    )
    
    if split_ratio is None:
        # Single dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        return dataloader
    else:
        # Split dataset
        train_ratio, val_ratio, test_ratio = split_ratio
        total_size = len(dataset)
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader


def create_synthetic_dataloader(num_samples: int = 1000,
                               batch_size: int = 8,
                               num_workers: int = 4,
                               num_frames: int = 16,
                               height: int = 256,
                               width: int = 256,
                               split_ratio: Optional[Tuple[float, float, float]] = None):
    """
    Create dataloader for synthetic video dataset.
    
    Args:
        num_samples: Number of synthetic samples
        batch_size: Batch size
        num_workers: Number of worker processes
        num_frames: Number of frames per video
        height: Video height
        width: Video width
        split_ratio: Train/val/test split ratios
    Returns:
        DataLoader or tuple of DataLoaders
    """
    # Create synthetic dataset
    dataset = SyntheticVideoDataset(
        num_samples=num_samples,
        num_frames=num_frames,
        height=height,
        width=width
    )
    
    if split_ratio is None:
        # Single dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        return dataloader
    else:
        # Split dataset
        train_ratio, val_ratio, test_ratio = split_ratio
        total_size = len(dataset)
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader


def create_mixed_dataloader(real_data_dir: str,
                           real_annotation_file: str,
                           synthetic_samples: int = 500,
                           real_weight: float = 0.7,
                           batch_size: int = 8,
                           num_workers: int = 4,
                           num_frames: int = 16,
                           height: int = 256,
                           width: int = 256,
                           split_ratio: Optional[Tuple[float, float, float]] = None):
    """
    Create dataloader that combines real and synthetic data.
    
    Args:
        real_data_dir: Directory containing real video files
        real_annotation_file: JSON file with real video annotations
        synthetic_samples: Number of synthetic samples
        real_weight: Weight for real data vs synthetic data
        batch_size: Batch size
        num_workers: Number of worker processes
        num_frames: Number of frames per video
        height: Video height
        width: Video width
        split_ratio: Train/val/test split ratios
    Returns:
        DataLoader or tuple of DataLoaders
    """
    # Create transforms
    transforms = get_video_transforms(
        height=height, 
        width=width, 
        augment=True
    )
    
    # Create real dataset
    real_dataset = VideoDataset(
        data_dir=real_data_dir,
        annotation_file=real_annotation_file,
        num_frames=num_frames,
        height=height,
        width=width,
        transform=transforms
    )
    
    # Create synthetic dataset
    synthetic_dataset = SyntheticVideoDataset(
        num_samples=synthetic_samples,
        num_frames=num_frames,
        height=height,
        width=width
    )
    
    # Create combined dataset
    combined_dataset = VideoTextDataset(
        datasets=[real_dataset, synthetic_dataset],
        weights=[real_weight, 1.0 - real_weight]
    )
    
    if split_ratio is None:
        # Single dataloader
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        return dataloader
    else:
        # Split dataset
        train_ratio, val_ratio, test_ratio = split_ratio
        total_size = len(combined_dataset)
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            combined_dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader 