import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj
from src.models.base_expert import GNNResidualBlock

class ZIGNNExpert(nn.Module):
    """
    5. Zero-Inflated GNN (The Intermittency & Sparsity Expert)
    
    Research Strategy: 
    This expert implements a 'Hurdle Model' within the GNN framework. 
    It explicitly decouples the Bernoulli process (Will a sale occur?) 
    from the Gamma/Poisson process (How much will be sold?). 
    
    Key for sub-0.5: Reducing the L2 penalty on non-sale days while 
    maintaining high-fidelity volume prediction.
    """
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 num_layers: int = 3, 
                 heads: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        
        # High-capacity shared backbone
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Shared Graph Representation Layers
        # These layers learn a common latent space that understands 
        # both stock-out risks and demand surges.
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // heads, 
                heads=heads, 
                concat=True
            )
            self.layers.append(GNNResidualBlock(hidden_dim, conv, dropout=dropout))
            
        # Task-Specific Decoders (The Research 'Secret Sauce')
        # Decoder 1: The 'Hurdle' (Classification Head)
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1) # Outputs Logits for Prob(Non-Zero)
        )
        
        # Decoder 2: The 'Magnitude' (Regression Head)
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim) # Outputs Latent Volume
        )
        
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Adj) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Adjacency (typically Behavioral or Structural) [2, E]
            
        Returns:
            volume_emb: The latent representation for the sales volume.
            prob_zero_logits: Logits representing the probability that 
                             actual sales > 0.
        """
        # 1. Project to Shared Latent Space
        x = self.input_proj(x)
        
        # 2. Multi-Layer Message Passing
        # Shared features are updated through neighborhood aggregation
        for layer in self.layers:
            x = layer(x, edge_index)
            
        x = self.final_norm(x)
        
        # 3. Dual-Head Decoding
        # Prob(Y > 0)
        prob_logits = self.prob_head(x) 
        
        # E(Y | Y > 0)
        volume_emb = self.vol_head(x)
        
        return volume_emb.contiguous(), prob_logits.contiguous()

    @staticmethod
    def get_mixture_logic():
        """Returns the mathematical Hurdle formulation used by this expert."""
        return r"P(Y=y) = \begin{cases} \pi & \text{if } y = 0 \\ (1-\pi) f(y; \mu, \sigma) & \text{if } y > 0 \end{cases}"