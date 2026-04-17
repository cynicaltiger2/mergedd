"""
M5 Supreme GNN Pipeline — Entry Point

DATA INTEGRITY POLICY
─────────────────────
All WRMSSE scores logged to wandb must come from a held-out VALIDATION set,
never from the training loader. The validation set covers strictly days after
`train_cutoff_day` (default: d_1885, leaving d_1886–d_1913 as the 28-day val window).
This is enforced by using `val_loader` in trainer.evaluate(), not `train_loader`.

See also: src/utils/graph_builder.py — graph topology is also cutoff-guarded.
"""

import os
import argparse
import torch
import yaml
import logging
import numpy as np
import wandb

# Project Imports
from src.models.decagon_ensemble import DecagonEnsemble
from src.boosting.lgbm_expert import SupremeLGBMExpert
from src.boosting.xgb_expert import XGBExpert
from src.engine.trainer import SupremeTrainer
from src.engine.pipeline import M5SupremeDataset, M5DataEngine
from src.utils.metrics import WRMSSEMetric

# Setup Research-Grade Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("M5_Supreme_Orchestrator")


def _build_dataset(cfg_data: dict, mode: str) -> M5SupremeDataset:
    """Helper: builds a dataset for the given mode ('train' or 'val')."""
    x_key = f'x_{mode}'
    y_key = f'y_{mode}'
    if x_key not in cfg_data or y_key not in cfg_data:
        raise KeyError(
            f"Config missing data paths for mode='{mode}': "
            f"expected keys '{x_key}' and '{y_key}' under data:. "
            f"A proper train/val temporal split is REQUIRED to prevent data leakage. "
            f"See DATA INTEGRITY POLICY in this file."
        )
    return M5SupremeDataset(
        x_path=cfg_data[x_key],
        y_path=cfg_data[y_key],
        graph_dir=cfg_data['graph_dir'],
        meta_path=cfg_data['meta_path'],
        mode=mode,
    )


def run_supreme_pipeline(config_path: str, hawkes_augmentation: bool = False):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # ─── Hawkes Chaos Config (loaded only when active) ────────────────
    chaos_cfg = None
    if hawkes_augmentation:
        from src.chaos.chaos_config import _load_config as load_chaos_config
        chaos_cfg = load_chaos_config()
        logger.info("Hawkes Augmentation ENABLED — chaos_config.yaml loaded")
        logger.info(f"  Hawkes defaults: α={chaos_cfg['hawkes']['default_alpha']}, "
                     f"β={chaos_cfg['hawkes']['default_beta']}")

    # Initialize Experiment Tracking
    wandb.init(project="M5_Decagon_Ensemble", config=cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Execution Engine Initialized on: {device} (A100 80GB Optimized)")

    # 2. Data Engineering & Stream Pipeline
    # ──────────────────────────────────────────────────────────────────────────
    # LEAKAGE FIX (L1): We build SEPARATE train and val datasets.
    #   train_ds → days d_1 … d_1885   (training window)
    #   val_ds   → days d_1886 … d_1913 (28-day held-out validation window)
    #
    # The val_loader is NEVER passed to train_epoch(). The WRMSSE logged to
    # wandb always comes from evaluate(val_loader) — never from evaluate(train_loader).
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("Loading training dataset...")
    train_ds = _build_dataset(cfg['data'], mode='train')
    train_engine = M5DataEngine(train_ds, batch_size=cfg['train']['batch_size'], workers=12)
    train_loader = train_engine.loader

    logger.info("Loading validation dataset (held-out temporal split)...")
    val_ds = _build_dataset(cfg['data'], mode='val')
    val_engine = M5DataEngine(val_ds, batch_size=cfg['train']['batch_size'], workers=4)
    val_loader = val_engine.loader

    # 3. Model Architecture Construction
    model = DecagonEnsemble(
        in_dim=cfg['model']['in_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        sig_edge_dim=cfg['model']['sig_dim']
    ).to(device)

    # Advanced Optimizer with Decoupled Weight Decay (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay']
    )

    # OneCycleLR: Best for escaping local minima in deep GNNs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg['train']['lr'],
        steps_per_epoch=len(train_loader),
        epochs=cfg['train']['epochs']
    )

    # 4. Supreme Trainer Initialization (VAT + EMA enabled)
    trainer = SupremeTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        weights=train_ds.weights,
        scale=train_ds.scales,
        vat_epsilon=cfg['train']['vat_eps'],
        ema_decay=cfg['train']['ema_decay']
    )

    # 5. Multi-Phase Training Protocol
    logger.info("Starting Phase 1: Expert Latent Representation Learning...")
    best_val_wrmsse = float('inf')

    for epoch in range(cfg['train']['epochs']):
        # ── Train ──
        metrics = trainer.train_epoch(train_loader)

        # ── Validate on HELD-OUT data — never on train_loader ──────────────
        # LEAKAGE FIX (L1): val_loader covers strictly post-cutoff days.
        # The WRMSSE reported here is a genuine out-of-sample score.
        val_wrmsse = trainer.evaluate(val_loader)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {metrics['total_loss']:.4f} | "
            f"Val WRMSSE: {val_wrmsse:.4f}"    # <-- this is now a real validation score
        )

        log_data = {**metrics, "val_wrmsse": val_wrmsse, "epoch": epoch}

        # ─── Hawkes Robustness Logging ───────────────────────────────────────
        if hawkes_augmentation:
            robustness = trainer.evaluate_robustness(val_loader, wrmsse_clean=val_wrmsse)
            logger.info(f"Epoch {epoch:03d} | Robustness R = {robustness['robustness_R']:.4f}")
            log_data.update({
                "wrmsse_chaos":   robustness['wrmsse_chaos'],
                "robustness_R":   robustness['robustness_R'],
            })

        wandb.log(log_data)

        # Save checkpoint when validation WRMSSE improves
        if val_wrmsse < best_val_wrmsse:
            best_val_wrmsse = val_wrmsse
            trainer.save_checkpoint("checkpoints/best_model_v1.pt")
            logger.info(f"  ✓ New best val WRMSSE: {best_val_wrmsse:.4f} — checkpoint saved.")

    logger.info(f"Phase 1 complete. Best Val WRMSSE: {best_val_wrmsse:.4f}")

    # 6. Hybrid Fusion Expert Integration (LGBM + XGB)
    logger.info("Starting Phase 2: Hybrid Boosting Fusion...")

    # Generate GNN embeddings for tree-based models
    # FIX (Q6): model() returns (forecast, zi_logits, trust_weights) — 3 values.
    # Previously only 2 were unpacked, causing ValueError.
    with torch.no_grad():
        gnn_forecast, _zi_logits, _trust_weights = model(
            train_ds.x.to(device), train_ds.adj_matrices
        )

    # LEAKAGE FIX (L5): Both boosting experts now receive proper train/val splits.
    # Previously fit() was called with only training args, skipping the val split
    # required for early stopping, and the call would crash due to missing positional args.
    boosting_data = cfg.get('boosting', {})
    required_boosting_keys = ['x_train', 'y_train', 'x_val', 'y_val']
    missing = [k for k in required_boosting_keys if k not in boosting_data]
    if missing:
        logger.error(
            f"Boosting config missing keys: {missing}. "
            "Skipping Phase 2. Add x_val and y_val paths to configs/supreme_config.yaml."
        )
    else:
        lgbm_expert = SupremeLGBMExpert()
        lgbm_expert.fit(
            x_train=boosting_data['x_train'],
            y_train=boosting_data['y_train'],
            x_val=boosting_data['x_val'],
            y_val=boosting_data['y_val'],
            train_weights=train_ds.weights.numpy(),
            val_weights=val_ds.weights.numpy(),
        )

        xgb_expert = XGBExpert()
        xgb_expert.fit(
            x_train=boosting_data['x_train'],
            y_train=boosting_data['y_train'],
            x_val=boosting_data['x_val'],
            y_val=boosting_data['y_val'],
            train_weights=train_ds.weights.numpy(),
            val_weights=val_ds.weights.numpy(),
        )

        # 7. Final Weighted Blending
        gnn_final  = gnn_forecast.cpu().numpy()
        lgbm_final = lgbm_expert.predict(boosting_data['x_test'])
        xgb_final  = xgb_expert.predict(boosting_data['x_test'])

        # Weights: 0.6 GNN / 0.2 LGBM / 0.2 XGB — tune via Optuna for optimal blend
        final_prediction = (0.6 * gnn_final) + (0.2 * lgbm_final) + (0.2 * xgb_final)

        # 8. Post-Processing
        final_prediction = np.maximum(0, final_prediction)   # Sales can't be negative
        logger.info(f"Pipeline Complete. Best Val WRMSSE: {best_val_wrmsse:.4f}. "
                    "Final predictions clipped to [0, ∞).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M5 Supreme GNN Pipeline")
    parser.add_argument(
        '--hawkes-augmentation',
        action='store_true',
        default=False,
        help='Enable Hawkes process chaos augmentation during training. '
             'When absent, behaviour is 100%% identical to the original pipeline.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/supreme_config.yaml',
        help='Path to the main pipeline config YAML.'
    )
    args = parser.parse_args()

    run_supreme_pipeline(
        config_path=args.config,
        hawkes_augmentation=args.hawkes_augmentation,
    )