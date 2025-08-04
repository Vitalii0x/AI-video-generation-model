import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionModel(nn.Module):
    """Denoising diffusion probabilistic model for video generation."""
    
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-compute values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


class UNet3D(nn.Module):
    """3D UNet for video denoising."""
    
    def __init__(self, in_channels=3, out_channels=3, d_model=768):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        
        # Initial convolution
        self.input_conv = nn.Conv3d(in_channels, d_model, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Downsampling path
        self.down1 = nn.Sequential(
            nn.Conv3d(d_model, d_model * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d_model * 2),
            nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(d_model * 2, d_model * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d_model * 4),
            nn.SiLU()
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv3d(d_model * 4, d_model * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model * 4),
            nn.SiLU()
        )
        
        # Upsampling path
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(d_model * 4, d_model * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, d_model * 2),
            nn.SiLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(d_model * 2, d_model, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU()
        )
        
        # Output
        self.output_conv = nn.Conv3d(d_model, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, text_embeddings=None):
        """
        Args:
            x: Video tensor [batch_size, channels, frames, height, width]
            t: Timestep tensor [batch_size]
            text_embeddings: Text embeddings [batch_size, text_length, d_model]
        Returns:
            output: Denoised video tensor
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial convolution
        x = self.input_conv(x)
        
        # Add time embedding
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        
        # Middle
        middle = self.middle(d2)
        
        # Upsampling
        u2 = self.up2(middle)
        u1 = self.up1(u2)
        
        # Output
        output = self.output_conv(u1)
        
        return output 