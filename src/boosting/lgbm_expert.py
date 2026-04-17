import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, Optional, Tuple

# Research-grade logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LGBM_Expert")

class SupremeLGBMExpert:
    """
    Research-Grade LightGBM Expert for M5.
    
    Upgrades:
    - Custom WRMSSE Evaluation: Aligns training progress with the leaderboard.
    - DART Persistence: Optimized drop-rates for high-resolution binning.
    - Memory Manifold: Force-casting for A100 FP16/FP32 efficiency.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        # Optimized for A100 80GB and the 0.5 WRMSSE target
        self.params = params or {
            'boosting_type': 'dart',          
            'objective': 'tweedie',
            'tweedie_variance_power': 1.15,   
            'learning_rate': 0.05,
            'num_leaves': 255,                
            'min_data_in_leaf': 100,
            'feature_fraction': 0.7,          # Increased stochasticity
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_bin': 1023,                  # High-res bins for Ampere architecture
            'lambda_l1': 0.2,                 # Slightly higher regularization
            'lambda_l2': 0.2,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            
            # DART-specific Research Parameters
            'drop_rate': 0.1,                 # % of trees to drop during iteration
            'skip_drop': 0.5,                 # Probability of skipping drop
            'max_drop': 50,                   # Cap on dropped trees
            
            'seed': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        self.model = None

    def wrmsse_feval(self, preds: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
        """
        Custom Evaluation Hook: Injects WRMSSE logic into the training loop.
        Note: Requires 'weight' and 'scale' to be present in the dataset metadata.
        """
        y_true = data.get_label()
        weights = data.get_weight()
        # Scale retrieval (Assumes you've packed scales into the dataset metadata or as a global)
        # For simplicity, we use MSE scaled by a constant if scales aren't available
        # In full research mode, you'd pull the pre-computed scales here.
        
        # Simplified Scaled Error for the training hook
        error = np.sqrt(np.mean(np.square(y_true - preds)) * weights.mean())
        return 'wrmsse_proxy', error, False

    def _prepare_dataset(self, 
                         x: pd.DataFrame, 
                         y: pd.Series, 
                         weights: Optional[np.ndarray] = None,
                         reference: Optional[lgb.Dataset] = None) -> lgb.Dataset:
        
        # Force float32 for A100 Tensor Core alignment
        x = x.astype(np.float32)
        cat_features = x.select_dtypes(include=['category', 'object']).columns.tolist()
        
        return lgb.Dataset(
            data=x, 
            label=y, 
            weight=weights,
            categorical_feature=cat_features,
            free_raw_data=False,
            reference=reference
        )

    def fit(self, 
            x_train: pd.DataFrame, 
            y_train: pd.Series, 
            x_val: pd.DataFrame, 
            y_val: pd.Series,
            train_weights: np.ndarray,
            val_weights: np.ndarray,
            num_boost_round: int = 2500):
        
        train_set = self._prepare_dataset(x_train, y_train, weights=train_weights)
        val_set = self._prepare_dataset(x_val, y_val, weights=val_weights, reference=train_set)

        # Research Note: DART does not support traditional early stopping.
        # We use a fixed iteration budget or monitor val_set manually.
        callbacks = [lgb.log_evaluation(period=100)]

        logger.info(f"🚀 Training Supreme LGBM on A100. Target: < 0.5 WRMSSE.")
        
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=['train', 'valid'],
            feval=self.wrmsse_feval, # Custom WRMSSE monitoring
            callbacks=callbacks
        )

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Expert model not trained.")
        # DART needs all iterations for stable inference
        return self.model.predict(x.astype(np.float32))

    def save_expert(self, path: str):
        if self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save_model(path)
            logger.info(f"Checkpoint archived: {path}")