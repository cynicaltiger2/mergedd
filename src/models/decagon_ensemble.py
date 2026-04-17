"""
DecagonEnsemble — Top-level orchestrator for the 9-Expert + MetaBlender system.

Import Strategy
───────────────
Python module names cannot start with a digit, so the expert files (1_h_gnn.py,
2_c_gnn.py, etc.) are imported via importlib.util rather than standard `from .X
import Y` syntax. This avoids renaming the files on disk while keeping the import
logic explicit and auditable in one place.
"""

import torch
import torch.nn as nn
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Dynamic import helper
# ──────────────────────────────────────────────────────────────────────────────

def _load(filename: str, class_name: str):
    """
    Loads a class from a module file whose name starts with a digit.

    Args:
        filename:   Bare filename, e.g. '1_h_gnn.py'
        class_name: The class to extract, e.g. 'HGNNExpert'
    """
    path = Path(__file__).parent / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expert module not found: {path}\n"
            f"Expected class '{class_name}' in '{filename}'. "
            f"Check that all expert files are present in src/models/."
        )
    spec = importlib.util.spec_from_file_location(class_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

# ──────────────────────────────────────────────────────────────────────────────
# Load all expert classes
# ──────────────────────────────────────────────────────────────────────────────

HGNNExpert       = _load('1_h_gnn.py',      'HGNNExpert')
CGNNExpert       = _load('2_c_gnn.py',      'CGNNExpert')
GraphormerExpert = _load('3_graphormer.py', 'GraphormerExpert')
SigGNNExpert     = _load('4_sig_gnn.py',    'SigGNNExpert')
ZIGNNExpert      = _load('5_zi_gnn.py',     'ZIGNNExpert')
EGNNExpert       = _load('6_e_gnn.py',      'EGNNExpert')
CalGNNExpert     = _load('7_cal_gnn.py',    'CalGNNExpert')
FlowGNNExpert    = _load('8_flow_gnn.py',   'FlowGNNExpert')
VATGNNExpert     = _load('9_vat_gnn.py',    'VATGNNExpert')
MetaBlender      = _load('10_blender.py',   'MetaBlender')


# ──────────────────────────────────────────────────────────────────────────────
# DecagonEnsemble
# ──────────────────────────────────────────────────────────────────────────────

class DecagonEnsemble(nn.Module):
    """
    Decagon Ensemble (GNN-10 Orchestrator).

    Wraps all 9 GNN experts into a unified manifold and fuses their
    latent representations via the MetaBlender's cross-attention gate.

    adj_dict expected keys:
        'hierarchical'      → H-GNN  (Expert 1)
        'behavioral'        → C-GNN  (Expert 2)  + edge_attr_dict['behavioral']
        'global_transformer'→ Graphormer (Expert 3)
        'path_signature'    → Sig-GNN (Expert 4) + edge_attr_dict['sig']
        'zero_inflation'    → ZI-GNN  (Expert 5)
        'economic'          → E-GNN   (Expert 6) + edge_attr_dict['economic']
        'temporal_sync'     → Cal-GNN (Expert 7) + edge_attr_dict['temporal_sync']
        'logistics_flow'    → Flow-GNN(Expert 8) + edge_attr_dict['logistics_flow']  (optional)
        'adversarial'       → VAT-GNN (Expert 9)
    """

    def __init__(self, in_dim: int, hidden_dim: int, sig_edge_dim: int = 16):
        super().__init__()

        # 1. The 9 Specialised GNN Experts
        self.h_gnn      = HGNNExpert(in_dim, hidden_dim)
        self.c_gnn      = CGNNExpert(in_dim, hidden_dim)
        self.graphormer = GraphormerExpert(in_dim, hidden_dim)
        self.sig_gnn    = SigGNNExpert(in_dim, hidden_dim, sig_edge_dim)
        self.zi_gnn     = ZIGNNExpert(in_dim, hidden_dim)
        self.e_gnn      = EGNNExpert(in_dim, hidden_dim)
        self.cal_gnn    = CalGNNExpert(in_dim, hidden_dim)
        self.flow_gnn   = FlowGNNExpert(in_dim, hidden_dim)
        self.vat_gnn    = VATGNNExpert(in_dim, hidden_dim)

        # 2. The Meta-Blender (10th Brain)
        self.blender = MetaBlender(hidden_dim, num_experts=9)

    def forward(
        self,
        x: torch.Tensor,
        adj_dict: Dict[str, torch.Tensor],
        time_idx: Optional[torch.Tensor] = None,
        edge_attr_dict: Optional[Dict] = None,
        store_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:              Raw node features [N, in_dim]
            adj_dict:       Dict of graph views (see class docstring for keys)
            time_idx:       Current day index [N, 1] — required for CalGNN
            edge_attr_dict: Optional edge attributes dict
            store_mask:     Optional integer tensor [N] mapping each node to its
                            store ID, passed to CGNNExpert to prevent cross-store
                            context pollution. If None, falls back to global mean.

        Returns:
            final_forecast:      [N, 28] point forecasts
            prob_zero_logits:    [N, 1]  sparsity mask logits from ZI-GNN
            trust_weights:       [N, 9]  expert attention weights from MetaBlender
        """
        if edge_attr_dict is None:
            edge_attr_dict = {}

        # ── Run all 9 experts ─────────────────────────────────────────────────
        h1 = self.h_gnn(x, adj_dict['hierarchical'])

        # CGNNExpert now accepts an optional store_mask to prevent cross-store
        # context leakage when training on multi-store batches.
        h2 = self.c_gnn(
            x,
            adj_dict['behavioral'],
            edge_weight=edge_attr_dict.get('behavioral'),
            store_mask=store_mask,
        )

        h3 = self.graphormer(x, adj_dict['global_transformer'])

        h4 = self.sig_gnn(
            x,
            adj_dict['path_signature'],
            edge_sig=edge_attr_dict.get('sig'),
        )

        # ZI-GNN is unique: returns (volume_emb, sparsity_logits)
        h5_vol, zi_logits = self.zi_gnn(x, adj_dict['zero_inflation'])

        h6 = self.e_gnn(
            x,
            adj_dict['economic'],
            edge_attr=edge_attr_dict.get('economic'),
        )

        h7 = self.cal_gnn(
            x,
            adj_dict['temporal_sync'],
            edge_attr=edge_attr_dict.get('temporal_sync'),
            time_idx=time_idx,
        )

        h8 = self.flow_gnn(
            x,
            adj_dict['logistics_flow'],
            edge_attr=edge_attr_dict.get('logistics_flow'),
        )

        h9 = self.vat_gnn(x, adj_dict['adversarial'])

        # ── Blend all expert embeddings ───────────────────────────────────────
        expert_embeddings = [h1, h2, h3, h4, h5_vol, h6, h7, h8, h9]
        final_forecast, trust_weights = self.blender(expert_embeddings, zi_logits)

        return final_forecast, zi_logits, trust_weights

    @torch.no_grad()
    def predict_all(
        self,
        x: torch.Tensor,
        adj_dict: Dict[str, torch.Tensor],
        time_idx: Optional[torch.Tensor] = None,
        edge_attr_dict: Optional[Dict] = None,
        store_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """High-performance inference — returns only the final forecast."""
        self.eval()
        forecast, _, _ = self.forward(x, adj_dict, time_idx, edge_attr_dict, store_mask)
        return forecast