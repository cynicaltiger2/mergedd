import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import List, Tuple

class ExpertAttentionGate(nn.Module):
    """
    Research-grade Cross-Attentional Gate.
    Queries the 9 experts using a Learned Context Query to determine 
    dynamic weights (Trust Scores) per node.
    """
    def __init__(self, hidden_dim: int, num_experts: int, heads: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Multi-head attention to find relationships between experts
        self.q_proj = nn.Linear(hidden_dim, hidden_dim) # Context Query
        self.k_proj = nn.Linear(hidden_dim, hidden_dim) # Expert Keys
        self.v_proj = nn.Linear(hidden_dim, hidden_dim) # Expert Values
        
        self.temp_scale = nn.Parameter(torch.ones(1) * math.sqrt(hidden_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, context: Tensor, expert_stack: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            context: [N, hidden_dim] - Global node/time context
            expert_stack: [N, num_experts, hidden_dim] - Embeddings from all experts
        """
        N, E, H = expert_stack.shape
        
        # 1. Scaled Dot-Product Attention
        # Query: [N, 1, H], Key: [N, E, H]
        query = self.q_proj(context).unsqueeze(1)
        keys = self.k_proj(expert_stack)
        values = self.v_proj(expert_stack)
        
        # Attention Scores: [N, 1, E]
        attn_logits = torch.bmm(query, keys.transpose(1, 2)) / self.temp_scale
        attn_weights = F.softmax(attn_logits, dim=-1) # [N, 1, E]
        
        # 2. Weighted Sum of Expert Representations
        # [N, 1, E] * [N, E, H] -> [N, 1, H]
        blended_emb = torch.bmm(attn_weights, values).squeeze(1)
        
        return blended_emb, attn_weights.squeeze(1)

class MetaBlender(nn.Module):
    """
    10. The Attentional Meta-Blender (The Final Brain)
    
    Research Strategy: 
    This is a Heterogeneous Mixture of Experts. It takes the latent 
    representations from the 9 GNN experts and blends them using 
    contextual attention. It also integrates the Sparsity Mask from 
    the ZI-GNN to prevent 'drift' on zero-sale days.
    """
    def __init__(self, 
                 hidden_dim: int, 
                 num_experts: int = 9, 
                 forecast_horizon: int = 28):
        super().__init__()
        
        # 1. The Blender Core
        self.gate = ExpertAttentionGate(hidden_dim, num_experts)
        
        # 2. Contextual Query Generator
        # Combines the node's base features with a latent summary to 
        # decide which expert to trust.
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. Final Multi-Step Forecast Regressor
        # Maps the blended embedding to the 28-day forecast horizon.
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, forecast_horizon) 
        )

    def forward(self, 
                expert_embeddings: List[Tensor], 
                prob_zero_logits: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            expert_embeddings: List of 9 [N, hidden_dim] tensors.
            prob_zero_logits: [N, 28] logits from ZI-GNN Expert.
            
        Returns:
            final_forecast: [N, 28] point forecasts.
            trust_weights: [N, 9] weights assigned to each expert.
        """
        # 1. Construct the Expert Tensor Stack
        # Shape: [N, 9, hidden_dim]
        expert_stack = torch.stack(expert_embeddings, dim=1)
        
        # 2. Generate Context Query
        # We use the average of experts as a 'Base Context'
        base_context = torch.mean(expert_stack, dim=1)
        query_context = self.context_net(base_context)
        
        # 3. Attentional Blending
        # blended_emb: [N, hidden_dim], weights: [N, 9]
        blended_emb, trust_weights = self.gate(query_context, expert_stack)
        
        # 4. Regression to 28-day horizon
        # Outputs Log-Volume for Tweedie stability
        log_volume = self.regressor(blended_emb)
        
        # 5. Zero-Inflation Gating (The Hurdle)
        # We apply the sparsity mask to the volume.
        # Forecast = Volume * Prob(Non-Zero)
        prob_not_zero = torch.sigmoid(-prob_zero_logits) # Logit to Prob(Non-Zero)
        final_forecast = torch.exp(log_volume) * prob_not_zero
        
        return final_forecast, trust_weights

    @staticmethod
    def get_blender_logic():
        """Returns the MoE trust formulation."""
        return r"Y_hat = [ \sum w_i(context) * E_i ] * Mask_sparsity"