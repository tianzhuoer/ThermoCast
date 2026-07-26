"""
统一特征提取工具：为 TimesFM、Chronos-2、LSTM、Transformer 构建 (K, 8) 特征张量。

特征定义（每步 8 维，与 Qwen3 文本输入的信息完全对应）：
  0  depth_norm  — AUV 深度归一化到 [-1, 1]（TSF_DEPTH_MIN/MAX）
  1  T_grad      — 温度剖面最大绝对梯度（跨 K 步 z-score）
  2  avg_T       — 温度剖面均值（跨 K 步 z-score）
  3  S_grad      — 盐度剖面最大绝对梯度（跨 K 步 z-score）
  4  avg_S       — 盐度剖面均值（跨 K 步 z-score）
  5  season      — 季节 ID 归一化到 [0, 1]（season_id / 3）
  6  doy_sin     — 年内日期周期正弦
  7  doy_cos     — 年内日期周期余弦
"""

import numpy as np
import torch

from configs.shared import TSF_DEPTH_MIN, TSF_DEPTH_MAX

_D_RANGE = TSF_DEPTH_MAX - TSF_DEPTH_MIN
N_FEATURES = 8


def _norm_depth(d: float) -> float:
    return (d - TSF_DEPTH_MIN) / _D_RANGE * 2.0 - 1.0


def _step_stats(profile):
    vals = [v if np.isfinite(v) else 0.0 for v in (float(x) for x in profile)]
    if len(vals) < 2:
        return 0.0, float(vals[0]) if vals else 0.0
    max_grad = max(abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1))
    return max_grad, sum(vals) / len(vals)


def _zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return (arr - arr.mean()) / (arr.std() + eps)


def make_feature_tensor(ex) -> torch.Tensor:
    """
    将单个 CTDTSFDataset 样本转为 (K, 8) float32 特征张量。
    可直接 stack 成 (B, K, 8) 批次张量。
    """
    K         = len(ex['input_features'])
    sal_feats = ex.get('input_sal_features', [])
    has_sal   = len(sal_feats) == K

    depth_norm = np.array([_norm_depth(float(d)) for d in ex['input_depths']],
                          dtype=np.float32)

    T_grads, avg_Ts, S_grads, avg_Ss = [], [], [], []
    for t in range(K):
        mg_t, av_t = _step_stats(ex['input_features'][t])
        T_grads.append(mg_t)
        avg_Ts.append(av_t)
        if has_sal:
            mg_s, av_s = _step_stats(sal_feats[t])
        else:
            mg_s, av_s = 0.0, 0.0
        S_grads.append(mg_s)
        avg_Ss.append(av_s)

    T_grads = _zscore(np.array(T_grads, dtype=np.float32))
    avg_Ts  = _zscore(np.array(avg_Ts,  dtype=np.float32))
    S_grads = _zscore(np.array(S_grads, dtype=np.float32))
    avg_Ss  = _zscore(np.array(avg_Ss,  dtype=np.float32))

    season  = float(ex['season_id']) / 3.0
    doy_sin = float(ex['doy_sin'])
    doy_cos = float(ex['doy_cos'])

    feat = np.stack([
        depth_norm,
        T_grads, avg_Ts,
        S_grads, avg_Ss,
        np.full(K, season,  dtype=np.float32),
        np.full(K, doy_sin, dtype=np.float32),
        np.full(K, doy_cos, dtype=np.float32),
    ], axis=1)  # (K, 8)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(feat, dtype=torch.float32)
