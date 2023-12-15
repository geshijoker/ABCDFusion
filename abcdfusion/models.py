from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchinfo import summary

import dgl
from dgl.nn import GraphConv

class LinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
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
    def __init__(self, in_channels: int, sizes: List[int], hidden_dim: int=None):
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
        
    def forward(self, x: Tensor)-> Tensor:
        for block in self.blocks:
            x = block(x)
        return x
    
class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, p: int=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.project = nn.Linear(in_channels, 1)
        
    def forward(self, x: Tensor)-> Tensor:
        x = self.dropout(x)
        x = self.project(x)
        return x
    
class Ensemble(nn.Module):
    def __init__(self, discriminator, *models):
        super().__init__()
        self.extractors = []
        for model in models:
            self.extractors.append(model)
        self.classifier = discriminator
        self.n_models = len(self.extractors)
        
    def forward(self, xs: List[Tensor])-> Tensor:
        assert len(xs) == self.n_models, f'There are {len(xs)} input tensors but only {self.n_models} feature extractors'
        features = []
        for x, model in zip(xs, self.extractors):
            features.append(model(x))
        x = torch.cat(features, dim=1)
        x = self.classifier(x)
        return x
    
# Define a GCN model that considers edge weights
class GCNBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        self.conv = GraphConv(in_feat, out_feats)
        
    def forward(self, g, x):
        x = torch.relu(self.conv(g, x))
        return x

class GCNWithEdgeWeights(nn.Module):
    def __init__(self, in_feats, hid_sizes, out_feats):
        super(GCNWithEdgeWeights, self).__init__()
        self.blocks = nn.ModuleList([
                GCNBlock(in_feats, hidden_sizes[0]),
                *[GCNBlock(sizes[i], sizes[i+1]) for i in range(len(hid_sizes)-1)]
            ])
        self.project = GraphConv(hid_sizes[-1], out_feats)

    def forward(self, g, features, edge_weights):
        g.edata['weight'] = edge_weights
        x = self.blocks(g, features)
        x = self.project(g, x)
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
