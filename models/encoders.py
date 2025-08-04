import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange, repeat


class TextEncoder(nn.Module):
    """CLIP-based text encoder for processing text descriptions."""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.max_length = max_length
        
        # Freeze CLIP parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
    def forward(self, text):
        """
        Args:
            text: List of text strings or tensor of tokenized text
        Returns:
            text_embeddings: Text embeddings [batch_size, max_length, hidden_dim]
        """
        if isinstance(text, list):
            # Tokenize text
            tokens = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokens.input_ids.to(self.text_encoder.device)
            attention_mask = tokens.attention_mask.to(self.text_encoder.device)
        else:
            input_ids = text
            attention_mask = torch.ones_like(input_ids)
            
        # Get text embeddings
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return text_outputs.last_hidden_state


class VideoEncoder(nn.Module):
    """3D CNN encoder for processing video frames."""
    
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512], 
                 kernel_sizes=[3, 3, 3, 3], strides=[1, 2, 2, 2]):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for hidden_dim, kernel_size, stride in zip(hidden_dims, kernel_sizes, strides):
            layer = nn.Sequential(
                nn.Conv3d(
                    current_channels, hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)
            current_channels = hidden_dim
            
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        """
        Args:
            x: Video tensor [batch_size, channels, frames, height, width]
        Returns:
            features: Encoded video features [batch_size, hidden_dim, frames, h, w]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal and spatial dimensions."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TemporalPositionalEncoding(nn.Module):
    """Temporal positional encoding for video frames."""
    
    def __init__(self, d_model, max_frames=100):
        super().__init__()
        self.d_model = d_model
        self.max_frames = max_frames
        
        # Create temporal encoding
        position = torch.arange(0, max_frames).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_frames, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, frames, features]
        Returns:
            x + temporal_encoding
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0) 