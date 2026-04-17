import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj
from src.models.base_expert import GNNResidualBlock

class ElasticityAttention(nn.Module):
    """
    Research-grade Elasticity Head.
    Specifically computes the 'Shock sensitivity' between two nodes
    based on their relative price delta.

    NOTE: This class is a legacy auxiliary head preserved for future
    per-edge elasticity scoring. It is not currently instantiated inside
    EGNNExpert — the main layers use GATv2Conv directly.
    """
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=dim,
            out_channels=dim // heads,
            heads=heads,
            edge_dim=2, # Edge features: [Relative_Price_Gap, Competitor_Discount_Ratio]
            concat=True,
            fill_value='mean'
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return self.conv(x, edge_index, edge_attr)

class EGNNExpert(nn.Module):
    """
    6. Economic Elasticity GNN (The Financial Resilience Expert)
    
    Research Strategy: 
    Nodes are connected based on Substitute/Complement relationships.
    This GNN captures the 'Cross-Price Elasticity of Demand' (CPED). 
    It is the primary defense against price volatility and competitor shocks.
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 3, 
                 heads: int = 12, 
                 dropout: float = 0.15):
        super().__init__()
        
        # Economic Feature Projection
        # Focuses on normalized price features, SNAP status, and promo flags
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Deep Elasticity Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // heads, 
                heads=heads, 
                edge_dim=2, # Delta-Price and Delta-Promo
                concat=True
            )
            self.layers.append(GNNResidualBlock(hidden_dim, conv, dropout=dropout))
            
        # Price-Shock Gate
        # Learns to dampen or amplify signal based on the magnitude of price change
        self.shock_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        """
        Args:
            x: Node features [N, in_dim] (Must include sell_price, price_max, price_min)
            edge_index: Economic adjacency (Substitute/Complement goods) [2, E]
            edge_attr: Price-diff attributes [E, 2] 
                       e.g., [log(price_i / price_j), promo_diff]
        """
        # 1. Project to Economic Latent Space
        x = self.input_proj(x)
        
        # 2. Elasticity-Aware Message Passing
        # GATv2 dynamically weighs neighbors based on how 'cheap' they are 
        # relative to the current node.
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            
        # 3. Dynamic Shock Dampening
        # If the model detects an extreme price anomaly, the gate modulates
        # the embedding to prevent gradient explosion.
        gate = self.shock_gate(x)
        x = x * gate
        
        return self.final_norm(x).contiguous()

    @staticmethod
    def compute_elasticity_edges(df_prices):
        """
        Mathematical utility for GraphBuilder:
        Identify cross-price elasticity by observing historical co-variance
        when one item's price changes and others remain constant.

        NOTE: This method is a stub. The implementation lives in
        M5GraphBuilder.build_elasticity_edges() in src/utils/graph_builder.py.
        Calling this directly raises NotImplementedError to prevent silent None returns.
        """
        raise NotImplementedError(
            "Use M5GraphBuilder.build_elasticity_edges() instead. "
            "This static stub is intentionally unimplemented here to avoid "
            "silent None returns from a pass statement."
        )