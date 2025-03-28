import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO
        d_k=q.size(-1)
        scores = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            scores=scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores,dim = -1)
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        batch_size = Q.size(0)

        # Linear -> (B, S, n_heads*d_model)
        Q_ = self.query_layers(Q).view(batch_size, -1, self.n_heads, self.d_model)
        K_ = self.key_layers(K).view(batch_size, -1, self.n_heads, self.d_model)
        V_ = self.value_layers(V).view(batch_size, -1, self.n_heads, self.d_model)

        # (B, S, n_heads, d_model) -> (B, n_heads, S, d_model)
        Q_ = Q_.transpose(1, 2)
        K_ = K_.transpose(1, 2)
        V_ = V_.transpose(1, 2)

        # scaled dot product attention
        context, attn = self.attention(Q_, K_, V_, mask=mask)

        # (B, n_heads, S, d_model) -> (B, S, n_heads*d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)

        # 마지막 fc 통과
        output = self.fc(context)

        return output