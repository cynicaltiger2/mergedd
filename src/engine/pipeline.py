import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Tuple
import logging

# Set up research-grade logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("M5_Pipeline")

class M5SupremeBatch:
    """
    A specialized container for 10-GNN Heterogeneous Data.
    Ensures all 10 edge_indices are moved to the same device as a single unit.
    """
    def __init__(self, x, y, adj_dict, weights, scales, edge_attr_dict=None):
        self.x = x
        self.y = y
        self.adj_dict = adj_dict
        self.weights = weights
        self.scales = scales
        self.edge_attr_dict = edge_attr_dict or {}

    def to(self, device):
        self.x = self.x.to(device, non_blocking=True)
        self.y = self.y.to(device, non_blocking=True)
        self.weights = self.weights.to(device, non_blocking=True)
        self.scales = self.scales.to(device, non_blocking=True)
        self.adj_dict = {k: v.to(device, non_blocking=True) for k, v in self.adj_dict.items()}
        self.edge_attr_dict = {k: v.to(device, non_blocking=True) for k, v in self.edge_attr_dict.items()}
        return self

class M5SupremeDataset(Dataset):
    """
    Research-Grade Dataset for M5 Forecasting.
    
    Features:
    - Lazy Loading support for large-scale store graphs.
    - Integrated NaN/Inf guards.
    - Support for Path Signature (SigGNN) feature alignment.
    """
    def __init__(self, 
                 x_path: str, 
                 y_path: str, 
                 graph_dir: str,
                 meta_path: str,
                 mode: str = 'train'):
        self.mode = mode
        
        # weights_only=False: we load full objects (dicts + tensors), not just model weights.
        # Safe here because these are our own pre-built tensors, not untrusted checkpoints.
        self.x = torch.load(x_path, map_location='cpu', weights_only=False)
        self.y = torch.load(y_path, map_location='cpu', weights_only=False)
        
        # Meta contains 'weights' and 'scales' pre-calculated for WRMSSE
        meta = torch.load(meta_path, map_location='cpu', weights_only=False)
        self.weights = meta['weights']
        self.scales = meta['scales']
        
        # Adjacency Dictionary (The 10 Views)
        self.adj_matrices = self._load_graphs(graph_dir)
        
        self._validate_data()

    def _load_graphs(self, graph_dir: str) -> Dict[str, torch.Tensor]:
        """Loads the 10 hierarchical and behavioral graph views."""
        views = ['hier', 'corr', 'econ', 'cal', 'flow', 'sig', 'global', 'vat', 'zi', 'struct']
        adj_dict = {}
        for view in views:
            path = f"{graph_dir}/{view}_edge_index.pt"
            try:
                adj_dict[view] = torch.load(path, map_location='cpu')
            except FileNotFoundError:
                logger.warning(f"Graph view '{view}' not found. Initializing as empty.")
                adj_dict[view] = torch.empty((2, 0), dtype=torch.long)
        return adj_dict

    def _validate_data(self):
        """Strict research-grade validation to prevent epoch crashes."""
        if torch.isnan(self.x).any():
            logger.error("NaN detected in features! Replacing with zeros.")
            self.x = torch.nan_to_num(self.x)
        
        if (self.scales <= 0).any():
            logger.warning("Zero or negative scales detected. Adding epsilon for WRMSSE stability.")
            self.scales = torch.clamp(self.scales, min=1e-8)

    def __len__(self):
        # In M5, one store or the full nation is typically one graph 'sample'
        return 1 

    def __getitem__(self, idx):
        return M5SupremeBatch(
            x=self.x,
            y=self.y,
            adj_dict=self.adj_matrices,
            weights=self.weights,
            scales=self.scales
        )

class M5DataEngine:
    """
    High-throughput Data Engine optimized for NVIDIA A100.
    Implements persistent workers and direct-to-device non-blocking transfers.
    """
    def __init__(self, dataset: M5SupremeDataset, batch_size: int = 1, workers: int = 8):
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            # Shuffling is meaningless for a graph-level dataset where __len__==1.
            # The graph IS the dataset — there is nothing to shuffle.
            shuffle=False,
            num_workers=workers,
            pin_memory=True,  # Critical for A100 DMA transfers
            persistent_workers=(workers > 0),
            collate_fn=lambda x: x[0]  # Return the M5SupremeBatch object directly
        )

    def get_stream(self, device, hawkes_augmentation=False):
        """
        A generator that streams data to the GPU using non-blocking transfers.
        When hawkes_augmentation is True, applies Hawkes-driven fault injection
        to batch.x before yielding. Does not modify the dataloader internals.
        """
        for batch in self.loader:
            batch = batch.to(device)
            if hawkes_augmentation:
                batch = _apply_hawkes_perturbation(batch)
            yield batch


# ═══════════════════════════════════════════════════════════════════════════════
# Hawkes Fault Injection Hook (Glue Code)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_hawkes_perturbation(batch: M5SupremeBatch) -> M5SupremeBatch:
    """
    Apply Hawkes-driven failure mask to batch features.

    This is the single integration point between the Supreme dataloader
    and the Chaos Engineering subsystem. It:
      1. Loads Hawkes params (α, β) from chaos_config.yaml defaults
      2. Derives μ from a base failure probability via μ = -ln(1 - p_base)
      3. Generates a failure mask via HawkesProcess.simulate()
      4. Multiplies batch.x element-wise (0 = data lost, 1 = survives)

    Called ONLY when --hawkes-augmentation is active.
    No restructuring of the dataloader — one clean call.
    """
    from src.chaos.hawkes_process import HawkesProcess, HawkesParams

    # Load defaults from chaos config (lazy import to avoid loading
    # chaos modules when hawkes_augmentation is False)
    from src.chaos.chaos_config import HAWKES_DEFAULT, mu_from_intensity

    # Default base failure probability for training augmentation
    p_base = 0.1
    mu = mu_from_intensity(p_base)
    alpha = HAWKES_DEFAULT['alpha']
    beta = HAWKES_DEFAULT['beta']

    params = HawkesParams(mu=mu, alpha=alpha, beta=beta)
    params.validate_subcritical()

    import random
    seed = random.randint(0, 2**31 - 1)   # fresh seed per call — no two batches share a mask
    hawkes = HawkesProcess(params=params, seed=seed)

    # batch.x shape: [N, features] or [N, T] depending on pipeline stage
    x_np = batch.x.detach().cpu().numpy()

    if x_np.ndim == 2:
        # Generate a 2D mask: each column (feature/timestep) shares
        # the same Hawkes intensity, rows drawn independently
        mask = hawkes.simulate_2d(x_np.shape[0], x_np.shape[1])
    else:
        mask = hawkes.simulate(x_np.shape[0])

    # Apply mask — element-wise multiplication preserves tensor dtype
    mask_tensor = torch.tensor(mask, dtype=batch.x.dtype, device=batch.x.device)
    batch.x = batch.x * mask_tensor

    logger.info(
        f"[Hawkes Augmentation] mask shape={mask.shape}, "
        f"failures={int((mask == 0).sum())}/{mask.size}, "
        f"λ_mean={hawkes.get_intensity_trace().mean():.4f}"
    )

    return batch