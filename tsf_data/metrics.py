"""
TSF 评估指标（在反归一化后的米空间计算）。

MAE   = (1/H) Σ|ŷ-y|
MSE   = (1/H) Σ(ŷ-y)²
RMSE  = √MSE
SMAPE = (200/H) Σ |ŷ-y| / (|y|+|ŷ|)          单位: %
MAPE  = (100/H) Σ |ŷ-y| / |y|                 单位: %
MASE  = per_sample_MAE / per_sample_naive_step  无量纲（naive: step-1 差分）
"""

import numpy as np


def compute_metrics(preds_m: np.ndarray, targets_m: np.ndarray) -> dict:
    """
    preds_m / targets_m: (N, H)  float, 单位：米（负深度）

    Returns
    -------
    dict with keys: mae, mse, rmse, smape, mape, mase, step_mae
    """
    err = preds_m - targets_m   # (N, H)

    mae  = float(np.abs(err).mean())
    mse  = float(np.square(err).mean())
    rmse = float(np.sqrt(mse))

    # SMAPE (%)
    smape = float(
        (200.0 * np.abs(err) / (np.abs(targets_m) + np.abs(preds_m) + 1e-8)).mean()
    )

    # MAPE (%)
    mape = float(
        (100.0 * np.abs(err) / (np.abs(targets_m) + 1e-8)).mean()
    )

    # MASE: s=1，逐样本计算再取均值
    H = targets_m.shape[1]
    if H > 1:
        naive_step = np.abs(np.diff(targets_m, axis=1)).mean(axis=1)   # (N,)
        per_mae    = np.abs(err).mean(axis=1)                           # (N,)
        mase       = float((per_mae / np.maximum(naive_step, 1e-8)).mean())
    else:
        mase = float("nan")

    step_mae = np.abs(err).mean(axis=0).tolist()

    return dict(mae=mae, mse=mse, rmse=rmse,
                smape=smape, mape=mape, mase=mase,
                step_mae=step_mae)
