import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention and feed-forward."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TemporalTransformer(nn.Module):
    """Temporal transformer for processing video sequences."""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, frames, features]
            mask: Optional mask for padding
        Returns:
            output: Transformed tensor [batch_size, frames, features]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SpatialTransformer(nn.Module):
    """Spatial transformer for processing spatial features."""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, height*width, features]
            mask: Optional mask for padding
        Returns:
            output: Transformed tensor [batch_size, height*width, features]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class CrossAttention(nn.Module):
    """Cross-attention between text and video features."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, video_features, text_features, mask=None):
        """
        Args:
            video_features: [batch_size, frames*h*w, d_model]
            text_features: [batch_size, text_length, d_model]
            mask: Optional mask for text padding
        Returns:
            attended_features: Cross-attended features
        """
        attended_features, _ = self.attention(
            video_features, text_features, text_features, mask
        )
        return self.norm(video_features + self.dropout(attended_features))


class TemporalCrossAttention(nn.Module):
    """Temporal cross-attention for video-text alignment."""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.self_attention_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, video_features, text_features, text_mask=None):
        """
        Args:
            video_features: [batch_size, frames, features]
            text_features: [batch_size, text_length, features]
            text_mask: Optional mask for text padding
        Returns:
            output: Cross-attended video features
        """
        for cross_attn, self_attn in zip(self.cross_attention_layers, self.self_attention_layers):
            # Cross-attention with text
            video_features = cross_attn(video_features, text_features, text_mask)
            # Self-attention on video features
            video_features = self_attn(video_features)
            
        return video_features 