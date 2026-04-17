import xgboost as xgb
import numpy as np
import pandas as pd
import os
import logging
import yaml
from typing import Dict, Optional, Tuple, Any

# Research-grade logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("XGB_Expert")

class XGBExpert:
    """
    Research-Grade XGBoost Expert for M5.
    
    Role in Decagon Ensemble: The Interaction Specialist.
    Focuses on high-depth trees to capture complex feature intersections 
    that GNNs and DART-LGBM might smooth over.
    """
    def __init__(self, config_path: str = "configs/boosting_config.yaml"):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            self.cfg = full_config['xgb']
            self.global_cfg = full_config['global']
            
        self.model = None
        self.params = self._prepare_params()

    def _prepare_params(self) -> Dict[str, Any]:
        """Optimizes parameters for A100 GPU and Tweedie distribution."""
        return {
            'objective': self.cfg['objective'],
            'tweedie_variance_power': self.cfg['tweedie_variance_power'],
            'tree_method': self.cfg['tree_method'], # 'gpu_hist' for A100
            'max_depth': self.cfg['max_depth'],     # Deep trees for interaction capture
            'learning_rate': self.cfg['learning_rate'],
            'subsample': self.cfg['subsample'],
            'colsample_bytree': self.cfg['colsample_bytree'],
            'alpha': self.cfg['alpha'],             # L1 Regularization
            'lambda': self.cfg['lambda'],           # L2 Regularization
            'predictor': self.cfg['predictor'],
            'random_state': self.global_cfg['random_state'],
            'verbosity': 0
        }

    def wrmsse_callback(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        """
        Research Evaluation Hook: Cross-validates based on M5 importance.
        """
        y_true = dtrain.get_label()
        weights = dtrain.get_weight()
        
        # Simplified proxy for WRMSSE gradient alignment
        # Root Mean Weighted Squared Error
        rmwse = np.sqrt(np.mean(weights * (y_true - preds)**2))
        return 'rmwse_proxy', rmwse

    def fit(self, 
            x_train: pd.DataFrame, 
            y_train: pd.Series, 
            x_val: pd.DataFrame, 
            y_val: pd.Series,
            train_weights: np.ndarray,
            val_weights: np.ndarray):
        
        logger.info("🚀 Initializing XGBoost Interaction Expert on A100...")

        # XGBoost DMatrix is more memory-efficient for large M5 tabular data
        # We ensure float32 to maximize A100 throughput
        dtrain = xgb.DMatrix(x_train.astype(np.float32), label=y_train, weight=train_weights)
        dval = xgb.DMatrix(x_val.astype(np.float32), label=y_val, weight=val_weights)

        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.global_cfg['num_boost_round'],
            evals=watchlist,
            early_stopping_rounds=self.global_cfg['early_stopping_rounds'],
            feval=self.wrmsse_callback,
            maximize=False,
            verbose_eval=100
        )
        
        logger.info(f"XGB Expert trained. Best Iteration: {self.model.best_iteration}")

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Interaction Expert not trained.")
        
        dtest = xgb.DMatrix(x.astype(np.float32))
        return self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))

    def save_expert(self, path: str):
        if self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save_model(path)
            logger.info(f"XGB Interaction Expert archived at {path}")

    def load_expert(self, path: str):
        self.model = xgb.Booster()
        self.model.load_model(path)
        return self