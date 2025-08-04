import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import random
from typing import Tuple, Optional


class VideoTransforms:
    """Transforms for video data augmentation and preprocessing."""
    
    def __init__(self, 
                 height: int = 256,
                 width: int = 256,
                 mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 augment: bool = True):
        """
        Args:
            height: Target video height
            width: Target video width
            mean: Normalization mean for each channel
            std: Normalization std for each channel
            augment: Whether to apply data augmentation
        """
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std
        self.augment = augment
        
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms to video tensor.
        
        Args:
            video: Video tensor [channels, frames, height, width]
        Returns:
            Transformed video tensor
        """
        # Resize if needed
        if video.shape[-2:] != (self.height, self.width):
            video = self._resize_video(video)
            
        # Data augmentation
        if self.augment:
            video = self._augment_video(video)
            
        # Normalize
        video = self._normalize_video(video)
        
        return video
        
    def _resize_video(self, video: torch.Tensor) -> torch.Tensor:
        """Resize video to target dimensions."""
        # video: [channels, frames, height, width]
        batch_size, channels, frames, height, width = video.shape
        
        # Reshape for interpolation: [channels * frames, height, width]
        video_reshaped = video.view(channels * frames, height, width)
        
        # Resize
        video_resized = F.interpolate(
            video_reshaped.unsqueeze(1),  # Add channel dimension
            size=(self.height, self.width),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # Remove channel dimension
        
        # Reshape back: [channels, frames, height, width]
        video_resized = video_resized.view(channels, frames, self.height, self.width)
        
        return video_resized
        
    def _augment_video(self, video: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to video."""
        # Random horizontal flip
        if random.random() > 0.5:
            video = torch.flip(video, dims=[-1])  # Flip width dimension
            
        # Random crop
        if random.random() > 0.5:
            video = self._random_crop(video)
            
        # Random brightness and contrast
        if random.random() > 0.5:
            video = self._adjust_brightness_contrast(video)
            
        # Random temporal jittering
        if random.random() > 0.5:
            video = self._temporal_jitter(video)
            
        return video
        
    def _random_crop(self, video: torch.Tensor, crop_ratio: float = 0.9) -> torch.Tensor:
        """Random crop video."""
        channels, frames, height, width = video.shape
        
        # Calculate crop size
        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)
        
        # Random crop position
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        
        # Crop video
        video_cropped = video[:, :, top:top + crop_height, left:left + crop_width]
        
        # Resize back to original size
        video_cropped = self._resize_video(video_cropped)
        
        return video_cropped
        
    def _adjust_brightness_contrast(self, video: torch.Tensor) -> torch.Tensor:
        """Adjust brightness and contrast randomly."""
        # Random factors
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        
        # Apply brightness
        video = video * brightness_factor
        video = torch.clamp(video, 0, 1)
        
        # Apply contrast
        mean = video.mean()
        video = (video - mean) * contrast_factor + mean
        video = torch.clamp(video, 0, 1)
        
        return video
        
    def _temporal_jitter(self, video: torch.Tensor) -> torch.Tensor:
        """Apply temporal jittering by randomly sampling frames."""
        channels, frames, height, width = video.shape
        
        # Randomly sample frames with replacement
        indices = torch.randint(0, frames, (frames,))
        video_jittered = video[:, indices, :, :]
        
        return video_jittered
        
    def _normalize_video(self, video: torch.Tensor) -> torch.Tensor:
        """Normalize video using mean and std."""
        # video: [channels, frames, height, width]
        mean = torch.tensor(self.mean).view(3, 1, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1, 1)
        
        video_normalized = (video - mean) / std
        
        return video_normalized


class VideoToTensor:
    """Convert video to tensor format."""
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        
    def __call__(self, video: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to tensor.
        
        Args:
            video: Video array [frames, height, width, channels]
        Returns:
            Video tensor [channels, frames, height, width]
        """
        # Convert to tensor
        video_tensor = torch.from_numpy(video).float()
        
        # Permute dimensions: [frames, height, width, channels] -> [channels, frames, height, width]
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        
        # Normalize to [0, 1] if needed
        if self.normalize and video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
            
        return video_tensor


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, video):
        for transform in self.transforms:
            video = transform(video)
        return video


def get_video_transforms(height: int = 256, 
                        width: int = 256, 
                        augment: bool = True,
                        normalize: bool = True):
    """Get standard video transforms."""
    transforms = []
    
    if normalize:
        transforms.append(VideoToTensor(normalize=True))
        
    transforms.append(VideoTransforms(height=height, width=width, augment=augment))
    
    return Compose(transforms) 