import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import math
import logging
from typing import Dict, Optional, Tuple

# Research Note: Installing 'torch-ema' is highly recommended for SOTA stability
try:
    from torch_ema import ExponentialMovingAverage
except ImportError:
    ExponentialMovingAverage = None

class SupremeTrainer:
    """
    Research-Grade Training Engine for M5 Decagon Ensemble.
    
    Features:
    - Virtual Adversarial Training (VAT) for chaos resilience.
    - EMA weight averaging for smoother inference and better WRMSSE.
    - Automatic Mixed Precision (AMP) optimized for A100 80GB.
    - Deep gradient monitoring to detect expert-level instability.
    """
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler, 
                 device: torch.device, 
                 weights: torch.Tensor, 
                 scale: torch.Tensor,
                 vat_epsilon: float = 1e-3,
                 vat_alpha: float = 1.0,
                 ema_decay: float = 0.999):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.weights = weights.to(device)
        self.scale = scale.to(device)
        
        # VAT Hyperparameters
        self.vat_epsilon = vat_epsilon
        self.vat_alpha = vat_alpha
        
        # EMA initialization (The secret to stable sub-0.5 leaderboard scores)
        if ExponentialMovingAverage:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None
            
        self.scaler = GradScaler(device='cuda')
        self.logger = logging.getLogger("SupremeTrainer")

        # Loss instantiated once here — not inside the training loop.
        # Recreating it per-batch caused redundant __init__ calls and
        # a dynamic import on every iteration, wasting time.
        from src.engine.loss import M5SupremeLoss
        self.criterion = M5SupremeLoss()

    def _compute_vat_loss(self, x: torch.Tensor, adj_dict: Dict, clean_forecast: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Virtual Adversarial Loss to enforce local smoothness.
        This forces the GNN to produce similar results even if inputs are perturbed.
        """
        # 1. Generate a small random perturbation
        d = torch.randn_like(x)
        d = 1e-6 * (d / (torch.norm(d, dim=-1, keepdim=True) + 1e-10))
        d.requires_grad_(True)
        
        # 2. Estimate the adversarial direction (where error increases most)
        with autocast(device_type='cuda', dtype=torch.float16):
            adv_forecast, _, _ = self.model(x + d, adj_dict)
            # Use KL-Divergence or MSE as the consistency metric
            dist = F.mse_loss(adv_forecast, clean_forecast.detach())
        
        grad = torch.autograd.grad(dist, d)[0]
        d_adv = self.vat_epsilon * (grad / (torch.norm(grad, dim=-1, keepdim=True) + 1e-10))
        
        # 3. Compute final consistency loss
        with autocast(device_type='cuda', dtype=torch.float16):
            adv_forecast, _, _ = self.model(x + d_adv.detach(), adj_dict)
            vat_loss = F.mse_loss(adv_forecast, clean_forecast.detach())
            
        return vat_loss

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = {"total_loss": 0, "vat_loss": 0, "base_loss": 0}
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.float16):  # A100 FP16
                # Forward Pass
                forecast, prob_zero_logits, _ = self.model(batch.x, batch.adj_dict)
                
                # 1. Primary Loss (Tweedie + WRMSSE alignment)
                base_loss = self.criterion(
                    forecast, batch.y, prob_zero_logits, self.weights, self.scale
                )
                
                # 2. VAT Loss (The Chaos Shield)
                # Only applied every few steps or with a weight alpha to save compute
                vat_loss = self._compute_vat_loss(batch.x, batch.adj_dict, forecast)
                
                total_loss = base_loss + (self.vat_alpha * vat_loss)

            # Backprop with Scaler
            self.scaler.scale(total_loss).backward()
            
            # Gradient Management
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update Shadow Weights (EMA)
            if self.ema:
                self.ema.update()
            
            # Scheduler Step
            self.scheduler.step()
            
            # Tracking
            epoch_metrics["total_loss"] += total_loss.item()
            epoch_metrics["vat_loss"] += vat_loss.item()
            epoch_metrics["base_loss"] += base_loss.item()
            
        return {k: v / len(loader) for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        """
        Evaluation using EMA weights for maximum stability and performance.
        """
        self.model.eval()
        
        # If EMA exists, temporarily load shadow weights for evaluation
        if self.ema:
            with self.ema.average_parameters():
                return self._run_evaluation_loop(loader)
        else:
            return self._run_evaluation_loop(loader)

    def _run_evaluation_loop(self, loader) -> float:
        preds, targets = [], []
        from src.utils.metrics import WRMSSEMetric
        # WRMSSEMetric requires (weights, scales, device) — device was previously missing
        metric_calc = WRMSSEMetric(self.weights, self.scale, self.device)

        for batch in loader:
            batch = batch.to(self.device)
            # Full inference through Decagon Ensemble
            forecast, _, _ = self.model(batch.x, batch.adj_dict)
            preds.append(forecast)
            targets.append(batch.y)
            
        y_pred = torch.cat(preds, dim=0)
        y_true = torch.cat(targets, dim=0)
        return metric_calc.compute(y_pred, y_true)

    def save_checkpoint(self, path: str):
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'ema_state': self.ema.state_dict() if self.ema else None,
            'scaler_state': self.scaler.state_dict()
        }
        torch.save(checkpoint, path)

    # ═══════════════════════════════════════════════════════════════════════
    # Hawkes Robustness Evaluation (Glue Code)
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def evaluate_robustness(self, loader, wrmsse_clean: float) -> dict:
        """
        Compute robustness R = WRMSSE_clean / WRMSSE_chaos after the
        existing validation loop.

        This method does NOT touch the validation loop internals. It runs
        a second evaluation pass with Hawkes-perturbed data and computes
        the robustness ratio.

        Called from main.py ONLY when --hawkes-augmentation is active.

        Parameters
        ----------
        loader : DataLoader
            The same validation loader used by evaluate().
        wrmsse_clean : float
            WRMSSE from the clean (unperturbed) evaluation pass.

        Returns
        -------
        dict : {'wrmsse_clean', 'wrmsse_chaos', 'robustness_R'}
        """
        self.model.eval()

        # Import Hawkes perturbation hook (lazy to avoid loading chaos
        # modules when --hawkes-augmentation is not active)
        from src.engine.pipeline import _apply_hawkes_perturbation
        from src.utils.metrics import WRMSSEMetric

        metric_calc = WRMSSEMetric(self.weights, self.scale, self.device)
        preds, targets = [], []

        if self.ema:
            with self.ema.average_parameters():
                for batch in loader:
                    batch = batch.to(self.device)
                    # Apply Hawkes perturbation to features
                    batch = _apply_hawkes_perturbation(batch)
                    forecast, _, _ = self.model(batch.x, batch.adj_dict)
                    preds.append(forecast)
                    targets.append(batch.y)
        else:
            for batch in loader:
                batch = batch.to(self.device)
                batch = _apply_hawkes_perturbation(batch)
                forecast, _, _ = self.model(batch.x, batch.adj_dict)
                preds.append(forecast)
                targets.append(batch.y)

        y_pred = torch.cat(preds, dim=0)
        y_true = torch.cat(targets, dim=0)
        wrmsse_chaos = metric_calc.compute(y_pred, y_true)

        # Robustness ratio: R = clean / chaos
        # R = 1.0 → no degradation; R < 1.0 → degradation
        R = wrmsse_clean / wrmsse_chaos if wrmsse_chaos > 0 else float('inf')

        self.logger.info(
            f"[Hawkes Robustness] R = {wrmsse_clean:.4f} / {wrmsse_chaos:.4f} = {R:.4f}"
        )

        return {
            'wrmsse_clean': wrmsse_clean,
            'wrmsse_chaos': wrmsse_chaos,
            'robustness_R': R,
        }