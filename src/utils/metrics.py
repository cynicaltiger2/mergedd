import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("M5_Metrics")

class WRMSSEMetric:
    """
    Research-Grade WRMSSE Evaluation Engine.
    
    This engine implements the M5 competition's primary metric. 
    It computes the Root Mean Squared Scaled Error across the 
    hierarchy and applies dollar-weighted importance.
    
    Optimized for: Vectorized execution on A100 GPU.
    """
    def __init__(self, 
                 weights: Tensor, 
                 scales: Tensor, 
                 device: torch.device):
        """
        Args:
            weights: The M5 dollar-weight for each series [N]
            scales: The historical scaling factor for each series [N]
                    (Average squared difference of consecutive days)
            device: The device for computation (cuda:0)
        """
        self.weights = weights.to(device)
        self.scales = scales.to(device)
        self.device = device
        self.eps = 1e-10

    @torch.no_grad()
    def compute(self, y_pred: Tensor, y_true: Tensor) -> float:
        """
        Calculates the WRMSSE score.
        
        Args:
            y_pred: Predicted sales [N, 28]
            y_true: Actual sales [N, 28]
        """
        # 1. Calculate Squared Error per time step
        # [N, 28]
        squared_error = (y_true - y_pred) ** 2
        
        # 2. Compute Mean Squared Error across the 28-day horizon
        # [N]
        mse_per_series = torch.mean(squared_error, dim=1)
        
        # 3. Calculate RMSSE (Root Mean Squared Scaled Error)
        # Scaled Error = MSE / Scale
        # RMSSE = sqrt(Scaled Error)
        rmsse = torch.sqrt(mse_per_series / (self.scales + self.eps))
        
        # 4. Apply Weights and Aggregate
        # WRMSSE = Sum(Weight_i * RMSSE_i)
        wrmsse = torch.sum(self.weights * rmsse)
        
        return wrmsse.item()

class HierarchicalAggregator:
    """
    Advanced utility to aggregate Level 12 (Item) predictions 
    up to Level 1 (Total).
    
    Essential for 'Supreme' validation as WRMSSE is technically 
    an average across all 12 hierarchical levels.
    """
    def __init__(self, aggregation_matrix: torch.sparse.FloatTensor):
        """
        Args:
            aggregation_matrix: A sparse matrix S [Total_Nodes, Item_Nodes]
                                where S * y_item = y_aggregate
        """
        self.S = aggregation_matrix

    def aggregate(self, y_item: Tensor) -> Tensor:
        """
        Vectorized summation up the Walmart hierarchy.
        """
        # If y_item is [30490, 28], returns [Total_Nodes, 28]
        return torch.sparse.mm(self.S, y_item)

def get_m5_weights_and_scales(
    sales_train_val: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    train_cutoff_day: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-computes M5 WRMSSE weights and scaling factors.

    LEAKAGE FIX (L4):
        The `scale` denominator must be computed ONLY from training-period sales
        (days d_1 … d_{train_cutoff_day}). Using the full dataset inflates the
        denominator and makes WRMSSE appear smaller (better) than reality.

    Args:
        sales_train_val:  M5 sales DataFrame (items × days, columns named 'd_N').
        prices:           M5 sell-prices DataFrame (for dollar-weight computation).
        calendar:         M5 calendar DataFrame (for SNAP / event alignment).
        train_cutoff_day: Last training day index. Only d_1 … d_{cutoff} are used
                          to compute the scale. MUST be set before any val/test days.

    Returns:
        weights: [N] dollar-weighted importance per series (normalised to sum=1).
        scales:  [N] mean absolute consecutive difference over TRAINING period only.
                 Series with zero variance (no sales ever) get scale = 1.0 to
                 prevent division by zero in WRMSSE.
    """
    # ── 1. Resolve training columns ──────────────────────────────────────────
    all_day_cols  = sorted(
        [c for c in sales_train_val.columns if c.startswith('d_')],
        key=lambda c: int(c.split('_')[1])
    )
    train_cols = [
        c for c in all_day_cols
        if int(c.split('_')[1]) <= train_cutoff_day
    ]
    if not train_cols:
        raise ValueError(
            f"No sales columns found for train_cutoff_day={train_cutoff_day}. "
            "Check that sales_train_val has 'd_N' columns."
        )

    train_sales = sales_train_val[train_cols].values.astype(np.float32)  # [N, T_train]

    # ── 2. Compute Scale (M5 Spec) ───────────────────────────────────────────
    # Scale_i = mean(|y_{i,t} - y_{i,t-1}|) for t = 2 … T_train
    # This is the denominator of RMSSE — the naïve in-sample one-step forecast error.
    consecutive_diff = np.abs(np.diff(train_sales, axis=1))       # [N, T_train-1]
    scales_np        = consecutive_diff.mean(axis=1)               # [N]
    # Guard: items that never sold (all zeros) produce scale=0 → division by zero.
    # Set to 1.0 for those items; their RMSSE will reflect raw MSE.
    scales_np = np.where(scales_np < 1e-8, 1.0, scales_np)
    scales    = torch.tensor(scales_np, dtype=torch.float32)

    # ── 3. Compute Weights ───────────────────────────────────────────────────
    # M5 weights are proportional to the dollar value of each series' sales
    # over the last 28 training days (the evaluation window immediately before cutoff).
    # If sell_price is not available per day, use mean price over the period.
    last_28_cols = train_cols[-28:]
    last_28_sales = sales_train_val[last_28_cols].values.astype(np.float32)  # [N, 28]

    # Try to look up sell_price per item; fall back to uniform weights if unavailable.
    try:
        item_ids    = sales_train_val.index
        item_prices = prices.loc[item_ids, 'sell_price'].values.astype(np.float32)
        dollar_sales = last_28_sales * item_prices[:, None]  # [N, 28]
    except (KeyError, AttributeError):
        logger.warning(
            "sell_price not found in prices DataFrame — "
            "falling back to unit-weight (unweighted) WRMSSE."
        )
        dollar_sales = last_28_sales

    total_dollar = dollar_sales.sum(axis=1)               # [N] — total $ per series
    total_sum    = total_dollar.sum()
    if total_sum < 1e-8:
        weights_np = np.ones(len(sales_train_val), dtype=np.float32) / len(sales_train_val)
    else:
        weights_np = total_dollar / total_sum              # Normalise to sum=1

    weights = torch.tensor(weights_np, dtype=torch.float32)

    logger.info(
        f"WRMSSE weights & scales computed on {len(train_cols)} training days "
        f"(d_1 → d_{train_cutoff_day}). Scale range: [{scales.min():.4f}, {scales.max():.4f}]"
    )
    return weights, scales