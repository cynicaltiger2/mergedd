import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj
from src.models.base_expert import GNNResidualBlock

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: This file was previously an exact copy of 7_cal_gnn.py (CalGNNExpert).
# That was a critical bug — the 8th "expert" was completely redundant, giving
# the MetaBlender 0 additional information from the logistics domain.
#
# This is now a proper FlowGNNExpert modelling directed supply-chain flow:
#   Warehouse → Distribution Centre → Store
#
# The key insight: supply-chain flow is DIRECTED (goods travel one way) and
# has temporal periodicity (reorder cycles). Edge features encode:
#   [lead_time_norm, reorder_cycle_phase, replenishment_urgency]
# ──────────────────────────────────────────────────────────────────────────────


class ReplenishmentGate(nn.Module):
    """
    Research-grade Replenishment Gate.

    Models the 'urgency' of restocking: when a node's stock is critically low
    (high replenishment_urgency on edges), messages from upstream suppliers
    should be weighted more heavily. This gate modulates the aggregated message
    by the urgency signal, implementing a learned version of the economic
    'order-up-to' policy.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Urgency projection: maps hidden_dim → sigmoid gate
        self.urgency_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor, aggregated: Tensor) -> Tensor:
        """
        x:          Current node embedding  [N, dim]
        aggregated: Neighborhood message    [N, dim]
        Returns: gated update x ← x + Sigmoid(urgency_proj(aggregated)) * aggregated
        """
        gate = self.urgency_proj(aggregated)
        return x + gate * aggregated


class FlowGNNExpert(nn.Module):
    """
    8. Logistics Flow GNN (The Supply-Chain Physics Expert)

    Research Strategy:
    Models the DIRECTED flow of goods through the Walmart supply network:
        Warehouse → Distribution Centre → Store → Item

    The graph is inherently directed (upstream → downstream). This is different
    from the hierarchical graph (Expert 1) which captures categorical structure;
    here we capture the physical replenishment chain.

    Edge features [E, 3]:
        [0] lead_time_norm        — normalised lead time between nodes (0–1)
        [1] reorder_cycle_phase   — where in the reorder cycle the edge is
                                    (0 = just ordered, 1 = reorder point reached)
        [2] replenishment_urgency — current stock / safety_stock ratio, clipped [0, 2]

    Key for sub-0.5:
        Capturing stockout risk BEFORE it appears in sales helps the model
        anticipate zero-sale periods caused by supply failure rather than
        true zero demand.
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 num_layers: int = 4,
                 heads: int = 8,
                 dropout: float = 0.15):
        super().__init__()

        # Directional flow projection
        # We maintain asymmetry: in_dim might differ from hidden_dim.
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # Flow Attention Layers
        # edge_dim=3: [lead_time_norm, reorder_cycle_phase, replenishment_urgency]
        self.layers = nn.ModuleList()
        self.replen_gates = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                edge_dim=3,           # Supply-chain edge features
                concat=True,
                dropout=dropout,
                fill_value='mean',    # Robust to missing upstream connections
                add_self_loops=False  # No self-loops — flow is strictly directed
            )
            self.layers.append(GNNResidualBlock(hidden_dim, conv, dropout=dropout))
            # Each layer has a specialised replenishment gate to modulate urgency
            self.replen_gates.append(ReplenishmentGate(hidden_dim))

        # Stockout Risk Head
        # Auxiliary multi-task head: outputs P(stockout) in the next reorder window.
        # Reserved for future multi-task training where the blender can consume this
        # signal directly. Not called in the current forward() pass intentionally —
        # it will be wired in once the blender accepts auxiliary heads.
        self.stockout_risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:          Node features [N, in_dim]
            edge_index: Directed flow adjacency [2, E]
                        Row 0 = source (upstream), Row 1 = target (downstream).
            edge_attr:  Supply-chain edge features [E, 3]
                        [lead_time_norm, reorder_cycle_phase, replenishment_urgency]
                        If None, GATv2 uses fill_value='mean' fallback.

        Returns:
            Latent node embedding with logistics-domain context [N, hidden_dim]
        """
        # 1. Project to Flow Latent Space
        x = self.input_proj(x)

        # 2. Directed Flow Message Passing
        # In each layer, information travels UPSTREAM → DOWNSTREAM.
        # An item at Store level 'hears' from its Distribution Centre about
        # incoming replenishment, reorder urgency, and lead time.
        for layer, gate in zip(self.layers, self.replen_gates):
            # Standard GNNResidualBlock handles pre-norm, DropPath, and FFN
            h = layer(x, edge_index, edge_attr=edge_attr)
            # Replenishment gate modulates the signal by urgency
            x = gate(x, h - x)  # Pass the residual as the 'aggregated message'

        # 3. Final Normalisation & Dropout
        x = self.final_norm(x)
        x = self.dropout(x)

        return x.contiguous()

    @staticmethod
    def construct_flow_edges(warehouse_df, _store_df=None, threshold_days: int = 14):
        """
        GraphBuilder utility: constructs directed Warehouse→Store edges.

        Args:
            warehouse_df: DataFrame with columns [warehouse_id, store_id, lead_time_days,
                         reorder_point, safety_stock, current_stock]
            _store_df:    Reserved for future store-level demand features (currently unused).
            threshold_days: Only add edges for warehouse-store pairs with lead_time ≤ threshold.

        Returns:
            edge_index [2, E], edge_attr [E, 3]
        """
        edges, attrs = [], []
        max_lead = threshold_days

        for _, row in warehouse_df.iterrows():
            if row['lead_time_days'] > threshold_days:
                continue
            lead_norm = row['lead_time_days'] / max_lead
            cycle_phase = (row['current_stock'] / (row['reorder_point'] + 1e-6)).clip(0, 1)
            urgency = (row['safety_stock'] / (row['current_stock'] + 1e-6)).clip(0, 2) / 2.0

            edges.append([row['warehouse_id'], row['store_id']])
            attrs.append([lead_norm, 1.0 - cycle_phase, urgency])

        if not edges:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3))

        return (
            torch.tensor(edges, dtype=torch.long).t().contiguous(),
            torch.tensor(attrs, dtype=torch.float)
        )