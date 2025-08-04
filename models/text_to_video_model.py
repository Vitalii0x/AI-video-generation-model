import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .encoders import TextEncoder, VideoEncoder, TemporalPositionalEncoding
from .transformer_blocks import TemporalTransformer, SpatialTransformer, TemporalCrossAttention
from .diffusion_model import DiffusionModel, UNet3D


class TextToVideoModel(nn.Module):
    """Main text-to-video generation model."""
    
    def __init__(self, 
                 d_model=768,
                 num_frames=16,
                 height=256,
                 width=256,
                 num_timesteps=1000,
                 num_heads=8,
                 num_layers=12,
                 d_ff=3072,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_timesteps = num_timesteps
        
        # Text encoder
        self.text_encoder = TextEncoder()
        
        # Video encoder
        self.video_encoder = VideoEncoder()
        
        # Temporal positional encoding
        self.temporal_pos_encoding = TemporalPositionalEncoding(d_model, num_frames)
        
        # Temporal transformer for video processing
        self.temporal_transformer = TemporalTransformer(
            d_model, num_heads, num_layers, d_ff, dropout
        )
        
        # Cross-attention between text and video
        self.cross_attention = TemporalCrossAttention(
            d_model, num_heads, num_layers, d_ff, dropout
        )
        
        # Spatial transformer for spatial features
        self.spatial_transformer = SpatialTransformer(
            d_model, num_heads, num_layers, d_ff, dropout
        )
        
        # Diffusion model
        self.diffusion = DiffusionModel(num_timesteps)
        
        # UNet for denoising
        self.unet = UNet3D(d_model=d_model)
        
        # Projection layers
        self.text_proj = nn.Linear(512, d_model)  # CLIP output to d_model
        self.video_proj = nn.Linear(self.video_encoder.output_dim, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 3)  # RGB output
        
    def encode_text(self, text):
        """Encode text descriptions."""
        text_embeddings = self.text_encoder(text)
        text_embeddings = self.text_proj(text_embeddings)
        return text_embeddings
        
    def encode_video(self, video):
        """Encode video frames."""
        # video: [batch_size, channels, frames, height, width]
        video_features = self.video_encoder(video)
        
        # Reshape for transformer: [batch_size, frames, height*width, features]
        batch_size, features, frames, h, w = video_features.shape
        video_features = video_features.permute(0, 2, 3, 4, 1)  # [B, F, H, W, C]
        video_features = video_features.reshape(batch_size, frames, h * w, features)
        
        # Project to d_model
        video_features = self.video_proj(video_features)
        
        return video_features
        
    def forward(self, video, text, t):
        """
        Forward pass for training.
        
        Args:
            video: Video tensor [batch_size, channels, frames, height, width]
            text: List of text strings
            t: Timestep tensor [batch_size]
        Returns:
            predicted_noise: Predicted noise for diffusion training
        """
        # Encode text
        text_embeddings = self.encode_text(text)
        
        # Encode video
        video_features = self.encode_video(video)
        
        # Add temporal positional encoding
        video_features = self.temporal_pos_encoding(video_features)
        
        # Apply temporal transformer
        video_features = self.temporal_transformer(video_features)
        
        # Apply cross-attention with text
        video_features = self.cross_attention(video_features, text_embeddings)
        
        # Reshape for spatial transformer: [batch_size, frames*h*w, features]
        batch_size, frames, hw, features = video_features.shape
        video_features = video_features.reshape(batch_size, frames * hw, features)
        
        # Apply spatial transformer
        video_features = self.spatial_transformer(video_features)
        
        # Reshape back to video format: [batch_size, features, frames, height, width]
        video_features = video_features.reshape(batch_size, frames, hw, features)
        video_features = video_features.permute(0, 3, 1, 2)  # [B, C, F, H*W]
        video_features = video_features.reshape(batch_size, features, frames, 
                                              int(hw**0.5), int(hw**0.5))
        
        # Apply UNet for denoising
        predicted_noise = self.unet(video_features, t, text_embeddings)
        
        return predicted_noise
        
    def sample(self, text, batch_size=1, num_frames=None, height=None, width=None):
        """
        Generate video from text.
        
        Args:
            text: List of text strings
            batch_size: Number of videos to generate
            num_frames: Number of frames (default: self.num_frames)
            height: Video height (default: self.height)
            width: Video width (default: self.width)
        Returns:
            video: Generated video tensor
        """
        num_frames = num_frames or self.num_frames
        height = height or self.height
        width = width or self.width
        
        device = next(self.parameters()).device
        
        # Encode text
        text_embeddings = self.encode_text(text)
        
        # Start from pure noise
        video = torch.randn(batch_size, 3, num_frames, height, width, device=device)
        
        # Denoising loop
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(video, text, t_tensor)
            
            # Update video (simplified denoising step)
            alpha_t = self.diffusion.alphas[t]
            alpha_t_cumprod = self.diffusion.alphas_cumprod[t]
            beta_t = self.diffusion.betas[t]
            
            # Remove predicted noise
            video = (video - beta_t * predicted_noise) / torch.sqrt(alpha_t)
            
            # Add noise for next step
            if t > 0:
                noise = torch.randn_like(video)
                video = video + torch.sqrt(beta_t) * noise
                
        return video
        
    def compute_loss(self, video, text, t):
        """Compute training loss."""
        # Add noise to video
        noise = torch.randn_like(video)
        noisy_video = self.diffusion.q_sample(video, t, noise)
        
        # Predict noise
        predicted_noise = self.forward(noisy_video, text, t)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss 