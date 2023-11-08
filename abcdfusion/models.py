from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchinfo import summary

class LinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
        )
        
    def forward(self, x: Tensor)-> Tensor:
        x = self.net(x)
        return x
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, emb_size: int):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Sequential(LinearBlock(in_channels, emb_size),
                                     nn.GELU())
        self.linear2 = LinearBlock(emb_size, in_channels)
        self.gelu = nn.GELU()
        
    def forward(self, x: Tensor):
        residual = x
        out = self.linear1(x)
        out = self.linear2(out)
        out += residual
        out = self.gelu(out)
        return out

class BinaryMLP(nn.Module):
    def __init__(self, in_channels: int, sizes: List[int], p: int=0.2, hidden_dim: int=None):
        super().__init__()
        if hidden_dim:
            self.blocks = nn.ModuleList([
                LinearBlock(in_channels, hidden_dim),
                *[ResidualBlock(hidden_dim, size) for size in sizes]
            ])
        else:
            self.blocks = nn.ModuleList([
                LinearBlock(in_channels, sizes[0]),
                *[LinearBlock(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
            ])
        self.dropout = nn.Dropout(p)
        if hidden_dim:
            self.project = nn.Linear(hidden_dim, 2)
        else:
            self.project = nn.Linear(sizes[-1], 2)
        
    def forward(self, x: Tensor)-> Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        x = self.project(x)
        return x
    
if __name__ == "__main__":

    in_channels = 10
    hidden_dim = 20
    sizes = [30, 30, 30]
    
    x = torch.rand(1, in_channels)

    model = BinaryMLP(in_channels, sizes, p=0.2, hidden_dim=hidden_dim)
    out = model(x)
    print('output shape', out.shape)
    summary(model, input_size=(1, in_channels))
