import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj
from src.models.base_expert import GNNResidualBlock

class BehavioralAttentionHead(nn.Module):
    """
    Research-grade Attention Head. 
    Implements multi-head attention over behavioral correlations 
    with specialized dropouts to prevent 'Attention Collapse'.
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=dim,
            out_channels=dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=1  # We use the correlation coefficient as an edge feature
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Tensor) -> Tensor:
        identity = x
        # Dynamic attention weights based on node features AND correlation strength
        x = self.conv(x, edge_index, edge_weight)
        return self.norm(x + identity)

class CGNNExpert(nn.Module):
    """
    2. Cross-Correlation Behavioral GNN (The Market Intelligence Expert)
    
    Research Strategy: 
    Nodes are connected based on historical Pearson Correlation of sales.
    This GNN uses dynamic attention to identify which 'neighbors' are 
    actual predictors vs. mere statistical noise.
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 3, 
                 heads: int = 16, 
                 dropout: float = 0.2):
        super().__init__()
        
        # High-dimensional embedding space projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Expert Layers: Each layer allows 'information' to jump across the store
        # between correlated items (e.g., Beer -> Charcoal).
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Using GATv2 for 'universal' attention ranking
            conv = GATv2Conv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // heads, 
                heads=heads, 
                edge_dim=1, # Uses correlation strength
                concat=True
            )
            self.layers.append(GNNResidualBlock(hidden_dim, conv, dropout=dropout))
            
        # Global Behavioral Context
        # Captures the 'average' behavior of the correlated cluster
        # NOTE: global_pool (AdaptiveAvgPool1d) was removed — it was declared
        # but never called in forward(), wasting ~hidden_dim^2 parameters.
        self.context_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Tensor,
                store_mask: Tensor = None) -> Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Correlation-based adjacency [2, E]
            edge_weight: The raw Pearson correlation values [E, 1]
            store_mask: Optional integer tensor [N] mapping each node to its store ID.
                        When provided, context is computed per-store so a signal from
                        a Texas store does not bleed into a California store's nodes.
                        When None, falls back to global mean (single-store graphs).
        """
        # 1. Project to Behavioral Latent Space
        x = self.input_proj(x)
        
        # 2. Dynamic Attention Message Passing
        # In this phase, nodes attend to 'Behavioral Neighbors'
        # weighted by their correlation strength.
        # edge_weight may be None if behavioral edge attrs weren't computed;
        # GNNResidualBlock handles the None case via its edge_attr=None path.
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_weight)
            
        # 3. Behavioral Context Injection
        # Computes a 'cluster context' per store so that individual items
        # understand whether their LOCAL store is trending — not a spurious
        # cross-state average. Falls back to global mean for single-store graphs.
        if store_mask is not None:
            # Per-store mean: inject only same-store context into each node.
            context = torch.zeros_like(x)
            for sid in store_mask.unique():
                sid_val = sid.item()  # convert 0-dim tensor → Python scalar for comparison
                mask = (store_mask == sid_val)
                store_mean = x[mask].mean(dim=0, keepdim=True)  # [1, H]
                gate = torch.sigmoid(self.context_gate(store_mean))  # [1, H]
                context[mask] = gate * store_mean
            x = x + context
        else:
            # Single-store graph fallback: global mean is the same as store mean.
            context = torch.mean(x, dim=0, keepdim=True)
            gate = torch.sigmoid(self.context_gate(context))
            x = x + (gate * context)
        
        return self.final_norm(x).contiguous()