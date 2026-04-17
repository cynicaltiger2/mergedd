import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torch import Tensor
from torch_geometric.nn import SAGEConv
from torch_geometric.typing import Adj
from src.models.base_expert import GNNResidualBlock

class GatedResidual(nn.Module):
    """
    Research-grade gating mechanism. 
    Instead of simple x + f(x), it learns how much of the new 
    information to 'let in' based on the current node state.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        g = self.gate(torch.cat([x, res], dim=-1))
        return x + g * res

class HGNNExpert(nn.Module):
    """
    1. Hierarchical-Spatial GNN (The Supreme Edition)
    
    Research Strategy: To achieve sub-0.5, the model must bridge the 
    geographic 'Physics' of Walmart with local product dynamics. 
    This version uses Gated GraphSAGE with Depth-4 reach.
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 4, 
                 dropout: float = 0.2):
        super().__init__()
        
        # High-capacity projection with Spectral-like stability
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Deep Hierarchy Layers
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        
        for i in range(num_layers):
            # We use SAGE with project=True for a more expressive learnable mapping
            conv = SAGEConv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim,
                normalize=True,
                project=True
            )
            # Incorporating GNNResidualBlock for standard stability
            self.layers.append(GNNResidualBlock(hidden_dim, conv, dropout=dropout))
            # Adding a secondary gating layer for research-grade flow control
            self.gates.append(GatedResidual(hidden_dim))
            
        # Final Hierarchical Context Aggregator
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialization logic (Kaiming Normal for GELU stability)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # trunc_normal_(std=0.02) is the standard for GELU/transformer-style
                # networks (ViT, DeiT, Graphormer). kaiming with 'relu' gain is
                # incorrect for GELU activations and over-scales initial weights.
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        N = Number of Items in the batch/graph
        edge_index must contain directed Item->Dept, Dept->Cat, Cat->Store, Store->State edges.
        """
        # 1. Feature Projection
        x = self.input_proj(x)
        
        # 2. Sequential Hierarchical Aggregation
        for i, layer in enumerate(self.layers):
            # Standard Message Passing
            h = layer(x, edge_index)
            # Gated Residual Update (Critical for sub-0.5)
            x = self.gates[i](x, h)
            
        # 3. Final Regularization
        x = self.final_norm(x)
        x = self.dropout(x)
        
        return x.contiguous()