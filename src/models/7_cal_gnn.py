import torch
import torch.nn as nn
from torch import Tensor
import math
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj
from src.models.base_expert import GNNResidualBlock

class FourierTemporalEncoding(nn.Module):
    """
    Research-grade Temporal Positional Encoding.
    Instead of standard embeddings, we use Fourier Basis functions to capture
    multiple seasonalities (Weekly, Monthly, Yearly) as continuous phases.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        # Pre-compute Fourier basis (Sine/Cosine)
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    def forward(self, t: Tensor) -> Tensor:
        """t: Time indices [N, 1]"""
        pe = torch.zeros(t.size(0), self.d_model, device=t.device)
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(t * self.div_term)
        pe[:, 1::2] = torch.cos(t * self.div_term)
        return pe

class CalGNNExpert(nn.Module):
    """
    7. Calendar/Event GNN (The Periodic Seasonality Expert)
    
    Research Strategy: 
    Nodes are connected if they share similar 'Temporal Fingerprints' 
    (e.g., items that all spike during SNAP days). 
    Uses Phase-Shift Attention to model how different events 'resonate' 
    across the store.
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 3, 
                 heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Fourier Encoding for the global calendar signal
        self.temporal_encoder = FourierTemporalEncoding(hidden_dim)
        
        # High-capacity feature projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Temporal Attention Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Using GATv2 to allow the model to attend to 'Seasonal Neighbors'
            # edge_dim=4 handles [Weekly_Sync, Monthly_Sync, Event_Intensity, Holiday_Flag]
            conv = GATv2Conv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // heads, 
                heads=heads, 
                edge_dim=4, 
                concat=True
            )
            self.layers.append(GNNResidualBlock(hidden_dim, conv, dropout=dropout))
            
        # Event gating: Allows the GNN to 'ignore' calendar signals on 
        # non-event days to prevent overfitting on noise.
        self.event_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, time_idx: Tensor) -> Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Temporal similarity adjacency [2, E]
            edge_attr: Temporal edge features [E, 4] (Weekly/Monthly correlation)
            time_idx: Current day index [N, 1]
        """
        # 1. Inject Global Fourier Seasonality
        # This gives every node a 'sense of time' before message passing
        temporal_bias = self.temporal_encoder(time_idx)
        x = self.input_proj(x) + temporal_bias
        
        # 2. Seasonal Resonance Message Passing
        # Information flows between nodes that react similarly to the calendar.
        # e.g., Beer and Snacks both 'vibrate' on Super Bowl Sunday.
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            
        # 3. Dynamic Event Gating
        # Modulates the signal based on whether today is a high-variance event.
        gate = self.event_gate(x)
        x = x * gate
        
        return self.final_norm(x).contiguous()

    @staticmethod
    def construct_temporal_edges(df_calendar, df_sales, threshold=0.7):
        """
        Utility for GraphBuilder: find products with same 'Event Elasticity'.

        NOTE: This static stub is intentionally unimplemented here.
        The full implementation is in M5GraphBuilder.build_temporal_sync_graph()
        in src/utils/graph_builder.py, which respects the train_cutoff_day
        data leakage policy.
        """
        raise NotImplementedError(
            "Use M5GraphBuilder.build_temporal_sync_graph() instead. "
            "That version correctly respects the training cutoff day "
            "to prevent data leakage."
        )