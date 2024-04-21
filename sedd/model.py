import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEDD(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.time_emb = TimestepEmbedder(config.model.hidden_size)
        self.pos_emb = PositionalEncoding(config.model.hidden_size, dropout=0.0)

        self.W_in = nn.Embedding(
            config.dataset.tokens + 1, 
            config.model.hidden_size,
        )
        self.W_out = nn.Linear(
            config.model.hidden_size, 
            config.dataset.tokens + 1, 
            bias=False
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=config.model.hidden_size, nhead=config.model.n_heads, batch_first=False)
                for _ in range(config.model.n_blocks)
            ]
        )
        

    def forward(self, x, t):
        assert len(x.shape) == 2
        
        # in
        x = self.W_in(x)
        t_emb = self.time_emb(t)[:, None, :]
        x = x + t_emb
        
        x = x.permute(1,0,2) # B, L, H -> L, B, H
        x = self.pos_emb(x)

        # mid
        for block in self.blocks:
            x = block(x)

        # out
        x = self.W_out(x)
        x = x.permute(1,0,2) # L, B, H -> B, L, H
        return x
    

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: torch.Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
