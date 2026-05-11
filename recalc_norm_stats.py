#!/usr/bin/env python3
"""
Recalculate normalization statistics from both train and test data.
"""

import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recalc_stats(data_root: str = 'a_dataset'):
    """Recalculate min/max stats across train + test splits."""
    
    data_root = Path(data_root)
    
    # Load all data
    logger.info("Loading train data...")
    X_train = np.load(data_root / 'train' / 'X_tmp.npy')
    te_train = np.load(data_root / 'train' / 'te_tmp.npy')
    ti_train = np.load(data_root / 'train' / 'ti_tmp.npy')
    na_train = np.load(data_root / 'train' / 'na_tmp.npy')
    ua_train = np.load(data_root / 'train' / 'ua_tmp.npy')
    
    logger.info("Loading test data...")
    X_test = np.load(data_root / 'test' / 'X_tmp.npy')
    te_test = np.load(data_root / 'test' / 'te_tmp.npy')
    ti_test = np.load(data_root / 'test' / 'ti_tmp.npy')
    na_test = np.load(data_root / 'test' / 'na_tmp.npy')
    ua_test = np.load(data_root / 'test' / 'ua_tmp.npy')
    
    # Concatenate train + test
    X = np.concatenate([X_train, X_test], axis=0)
    te = np.concatenate([te_train, te_test], axis=0)
    ti = np.concatenate([ti_train, ti_test], axis=0)
    na = np.concatenate([na_train, na_test], axis=0)
    ua = np.concatenate([ua_train, ua_test], axis=0)
    
    logger.info(f"Combined dataset sizes: X={X.shape}, te={te.shape}, na={na.shape}")
    
    # Compute stats
    new_stats = {}
    
    # X: linear scale
    new_stats['X_min'] = X.min(axis=0).astype(np.float32)
    new_stats['X_max'] = X.max(axis=0).astype(np.float32)
    logger.info(f"X: min={new_stats['X_min']}, max={new_stats['X_max']}")
    
    # te: log scale
    te_flat = te.ravel()
    te_pos = te_flat[te_flat > 0]
    new_stats['te_min'] = np.log(te_pos.min()).astype(np.float32)
    new_stats['te_max'] = np.log(te_pos.max()).astype(np.float32)
    new_stats['te_min_linear'] = te_pos.min().astype(np.float32)
    new_stats['te_max_linear'] = te_pos.max().astype(np.float32)
    logger.info(f"te (ln): min={new_stats['te_min']:.4f}, max={new_stats['te_max']:.4f}")
    
    # ti: log scale
    ti_flat = ti.ravel()
    ti_pos = ti_flat[ti_flat > 0]
    new_stats['ti_min'] = np.log(ti_pos.min()).astype(np.float32)
    new_stats['ti_max'] = np.log(ti_pos.max()).astype(np.float32)
    new_stats['ti_min_linear'] = ti_pos.min().astype(np.float32)
    new_stats['ti_max_linear'] = ti_pos.max().astype(np.float32)
    logger.info(f"ti (ln): min={new_stats['ti_min']:.4f}, max={new_stats['ti_max']:.4f}")
    
    # na: log scale per species
    na_log_min = []
    na_log_max = []
    for sp in range(na.shape[-1]):
        na_sp = na[..., sp].ravel()
        na_pos = na_sp[na_sp > 0]
        na_log_min.append(np.log(na_pos.min()))
        na_log_max.append(np.log(na_pos.max()))
    
    # Store LOG values in na_min/na_max (not linear!)
    new_stats['na_min'] = np.array(na_log_min, dtype=np.float32)
    new_stats['na_max'] = np.array(na_log_max, dtype=np.float32)
    # Also store linear versions for reference
    na_min_linear = []
    na_max_linear = []
    for sp in range(na.shape[-1]):
        na_sp = na[..., sp].ravel()
        na_pos = na_sp[na_sp > 0]
        na_min_linear.append(na_pos.min())
        na_max_linear.append(na_pos.max())
    new_stats['na_log_min'] = np.array(na_log_min, dtype=np.float32)
    new_stats['na_log_max'] = np.array(na_log_max, dtype=np.float32)
    logger.info(f"na (ln): min={new_stats['na_min']}, max={new_stats['na_max']}")
    
    # ua: linear scale — use percentiles (p2, p98) instead of min/max to handle outliers
    # This ensures the full data range [p2, p98] maps to [0, 1], with rare outliers clamped
    ua_p2 = []
    ua_p98 = []
    for sp in range(ua.shape[-1]):
        ua_sp = ua[..., sp].ravel()
        ua_p2.append(np.percentile(ua_sp, 2))
        ua_p98.append(np.percentile(ua_sp, 98))
    new_stats['ua_min'] = np.array(ua_p2, dtype=np.float32)
    new_stats['ua_max'] = np.array(ua_p98, dtype=np.float32)
    logger.info(f"ua (p2-p98): min={new_stats['ua_min']}, max={new_stats['ua_max']}")
    
    # Save
    out_path = data_root / 'norm_stats_minmax_recalc.npz'
    np.savez(out_path, **new_stats)
    logger.info(f"Saved recalculated stats to {out_path}")
    
    # Show comparison
    logger.info("\nComparison with original stats:")
    old_data = np.load(data_root / 'norm_stats_minmax.npz')
    
    print(f"  te_min: old={old_data['te_min']:.4e} -> new={new_stats['te_min']:.4e}")
    print(f"  te_max: old={old_data['te_max']:.4e} -> new={new_stats['te_max']:.4e}")
    print(f"  na_max[0]: old={old_data['na_max'][0]:.4e} -> new={new_stats['na_max'][0]:.4e}")
    print(f"  ua_min[0]: old={old_data['ua_min'][0]:.4e} -> new={new_stats['ua_min'][0]:.4e}")


if __name__ == '__main__':
    recalc_stats()
