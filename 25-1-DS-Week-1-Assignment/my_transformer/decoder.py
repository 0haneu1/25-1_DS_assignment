import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        #TODO
        # self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        # cross-attention
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        # Position-wise FF Net
        self.ff = FeedForwardLayer(d_model, d_ff)

        # layernorm
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        # 드롭아웃
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.dropout3 = DropoutLayer(dropout)
        # Residual
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
        self.residual3 = ResidualConnection()
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        # Masked self-attention
        x = self.norm1(self.residual1(x, lambda x: self.dropout1(self.self_attn(x, x, x, tgt_mask))))
        # 인코더-디코더 attention
        x = self.norm2(self.residual2(x, lambda x: self.dropout2(self.enc_dec_attn(x, memory, memory, src_mask))))
        # Feed Forward Network
        x = self.norm3(self.residual3(x, lambda x: self.dropout3(self.ff(x))))
        return x