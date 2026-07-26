import datetime
import hashlib
import os
import pickle
import random

import numpy as np
import scipy.io
from datasets import load_from_disk
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

import json

from configs.shared import (
    CTD_FOLDER_PATH, CTD_AUV_EXCLUDE_KEYWORDS,
    CTD_TSF_AUV_NOISE_STD, CTD_TSF_NEAR_THERMO_PROB,
    CTD_TSF_STRIDE, CTD_TSF_N_AUGMENT,
    DEPTH_MIN, DEPTH_MAX, SAMPLE_WINDOW_SIZE, STEP_SIZE,
    SPLIT_TRAIN_MOD, SPLIT_VAL_MOD, SPLIT_TEST_MOD,
    TSF_DATA_PATH,
    TSF_DEPTH_MIN, TSF_DEPTH_MAX,
    TSF_K, TSF_H,
    CTD_SPLIT_MANIFEST,
)

_SPLIT_MODS = {
    'train':      SPLIT_TRAIN_MOD,
    'validation': SPLIT_VAL_MOD,
    'test':       SPLIT_TEST_MOD,
}

# manifest split 名称 → CTDTSFDataset split 参数的映射
_MANIFEST_SPLIT_MAP = {
    'train':      ('llm_train',),
    'validation': ('llm_val', 'validation'),
    'test':       ('llm_test', 'test'),
}


def _load_manifest(manifest_path):
    """加载 split_manifest.json；不存在时返回 None。"""
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)['manifest']
    return None

# 缓存版本号：当样本构建逻辑变化时手动递增，避免复用旧缓存。
_DATASET_CACHE_VERSION = 3  # PO 改为按深度分 bin 取最新值


# ── 数据集磁盘缓存 ──────────────────────────────────────────────────────────────

def _dataset_cache_key(split, seed, ctd_folder, exclude_keywords, manifest_path=None):
    """哈希缓存键：覆盖全部影响样本内容的参数及 .mat 文件修改时间。"""
    h = hashlib.md5()
    params = (
        _DATASET_CACHE_VERSION,
        split, seed,
        TSF_K, TSF_H, CTD_TSF_STRIDE, CTD_TSF_N_AUGMENT,
        CTD_TSF_NEAR_THERMO_PROB, CTD_TSF_AUV_NOISE_STD,
        SAMPLE_WINDOW_SIZE, STEP_SIZE,
        DEPTH_MIN, DEPTH_MAX, TSF_DEPTH_MIN, TSF_DEPTH_MAX,
    )
    h.update(str(params).encode())
    h.update(str(sorted(exclude_keywords)).encode())
    all_files = sorted(f for f in os.listdir(ctd_folder) if f.endswith('.mat'))
    mat_files = [f for f in all_files if not any(kw in f for kw in exclude_keywords)]
    for fname in mat_files:
        mtime = os.path.getmtime(os.path.join(ctd_folder, fname))
        h.update(f"{fname}:{mtime:.3f}".encode())
    # manifest 内容纳入哈希：manifest 变化（或从无到有）时缓存自动失效
    _mp = manifest_path if manifest_path is not None else CTD_SPLIT_MANIFEST
    if _mp and os.path.exists(_mp):
        with open(_mp, 'rb') as f:
            h.update(f.read())
    else:
        h.update(b"no_manifest")
    return h.hexdigest()[:16]


# ── 辅助函数 ────────────────────────────────────────────────────────────────────

def _datenum2datetime(datenum):
    return (datetime.datetime.fromordinal(int(datenum))
            + datetime.timedelta(days=datenum % 1)
            - datetime.timedelta(days=366))


def _season_id(dt):
    m = dt.month
    if m in (12, 1, 2): return 0   # Winter
    if m in (3, 4, 5):  return 1   # Spring
    if m in (6, 7, 8):  return 2   # Summer
    return 3                        # Autumn


def _compute_thermo_depths(T_interp, depth_grid):
    """在物理合理深度带内取 |dT/dz| 最大处的深度（正深度，米）。"""
    n_time = T_interp.shape[1]
    thermo = np.empty(n_time)
    band_min = min(abs(TSF_DEPTH_MIN), abs(TSF_DEPTH_MAX))
    band_max = max(abs(TSF_DEPTH_MIN), abs(TSF_DEPTH_MAX))
    mid_depths = (depth_grid[:-1] + depth_grid[1:]) / 2.0
    band_mask = (mid_depths >= band_min) & (mid_depths <= band_max)
    if not band_mask.any():
        band_mask = np.ones_like(mid_depths, dtype=bool)

    for t in range(n_time):
        dz   = np.diff(depth_grid)
        grad = np.abs(np.diff(T_interp[:, t]) / np.where(dz == 0, np.nan, dz))
        grad = np.nan_to_num(grad, nan=-np.inf)
        band_indices = np.flatnonzero(band_mask)
        idx = int(band_indices[np.argmax(grad[band_mask])])
        thermo[t] = float(mid_depths[idx])
    return thermo


def _extract_features(profile_T, depth_grid, center_depth, n_samples, step_size):
    """在 center_depth 附近等间距插值采样 n_samples 个温度值，模拟 AUV 局部观测。"""
    half = (n_samples // 2) * step_size
    sample_depths = np.linspace(center_depth - half, center_depth + half, n_samples)
    sample_depths = np.clip(sample_depths, depth_grid.min(), depth_grid.max())
    f = interp1d(depth_grid, profile_T, kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    return f(sample_depths).tolist()


def _to_model_depth(depth):
    """统一使用负深度：海面 0 m，向下为负。"""
    return -abs(float(depth))


# ── 数据集类 ────────────────────────────────────────────────────────────────────

class TSFDataset(Dataset):
    """时序预测数据集（由 RL 仿真生成，按 episode_id % 10 划分 70/20/10）。"""

    def __init__(self, split='train', data_dir=None):
        if split not in _SPLIT_MODS:
            raise ValueError(f"split 必须是 'train'/'validation'/'test'，得到 '{split}'")
        data_dir = data_dir or TSF_DATA_PATH
        ds = load_from_disk(data_dir)
        sample0 = ds[0] if len(ds) > 0 else {}
        required = {'episode_id', 'input_depths', 'target_depths'}
        missing  = required - set(sample0.keys())
        if missing:
            raise KeyError(
                f"数据集缺少 TSF 必需字段: {missing}。\n"
                f"  当前字段: {sorted(sample0.keys())}\n"
                f"  数据路径: {data_dir}\n"
                "请先运行 DQNSeqPomdp_VLMdataGen.py 生成 TSF 数据集。"
            )
        mod_set = _SPLIT_MODS[split]
        self.samples = [s for s in ds if int(s['episode_id']) % 10 in mod_set]

    def __len__(self):   return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class CTDTSFDataset(Dataset):
    """
    直接从 CTD .mat 剖面时序滑窗切片构建 TSF 样本，无需 RL 仿真。

    split 划分由 split_manifest.json 文件级分配决定：
      train      → manifest['llm_train']
      validation → manifest['llm_val'] + manifest['validation']
      test       → manifest['llm_test'] + manifest['test']

    AUV 深度 = 温跃层深度 + N(0, CTD_TSF_AUV_NOISE_STD)，
    模拟 AUV 在温跃层附近搜索时的实际观测分布。
    """

    def __init__(self, split='train', ctd_folder=None, seed=42,
                 exclude_keywords=None, verbose=True, cache_dir=None,
                 manifest_path=None):
        if split not in _SPLIT_MODS:
            raise ValueError(f"split 必须是 'train'/'validation'/'test'，得到 '{split}'")
        ctd_folder       = ctd_folder or CTD_FOLDER_PATH
        exclude_keywords = exclude_keywords or CTD_AUV_EXCLUDE_KEYWORDS
        self._verbose    = verbose

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key        = _dataset_cache_key(split, seed, ctd_folder, exclude_keywords, manifest_path)
            cache_file = os.path.join(cache_dir, f"ctd_tsf_{split}_{key}.pkl")
            if os.path.exists(cache_file):
                if verbose:
                    print(f"[CTDTSFDataset] 从缓存加载 {split}  ({cache_file})")
                with open(cache_file, 'rb') as f:
                    self.samples = pickle.load(f)
                return
        else:
            cache_file = None

        rng      = np.random.default_rng(seed)
        self.samples = []

        # ── 根据 manifest 筛选本 split 对应的文件 ────────────────────────────
        _manifest_path = manifest_path if manifest_path is not None else CTD_SPLIT_MANIFEST
        manifest = _load_manifest(_manifest_path)
        allowed_tags = _MANIFEST_SPLIT_MAP[split]

        all_files = sorted(f for f in os.listdir(ctd_folder) if f.endswith('.mat'))
        if manifest is not None:
            mat_files = [f for f in all_files if manifest.get(f) in allowed_tags]
            if verbose:
                print(f"[CTDTSFDataset] split={split}  manifest 命中 {len(mat_files)} 个文件")
        else:
            # 无 manifest 时回退旧逻辑（按文件序号取模）
            mat_files = [f for f in all_files
                         if not any(kw in f for kw in exclude_keywords)]
            mod_set = _SPLIT_MODS[split]
            mat_files = [f for i, f in enumerate(mat_files) if (i % 10) in mod_set]
            if verbose:
                print(f"[CTDTSFDataset] 未找到 manifest，回退到取模划分，split={split}  {len(mat_files)} 个文件")

        for file_idx, fname in enumerate(mat_files):
            fpath = os.path.join(ctd_folder, fname)
            try:
                data = scipy.io.loadmat(fpath)
            except Exception:
                continue

            if not {'new_T_interp', 'new_S_interp', 'new_time_grid', 'depth_grid'}.issubset(data.keys()):
                continue

            T          = np.asarray(data['new_T_interp'])
            S          = np.asarray(data['new_S_interp'])
            time_grid  = np.asarray(data['new_time_grid']).flatten()
            depth_grid = np.asarray(data['depth_grid']).flatten()

            bad_cols = np.isnan(T).sum(axis=0) > 20
            T = T[:, ~bad_cols]; S = S[:, ~bad_cols]; time_grid = time_grid[~bad_cols]
            bad_rows = np.isnan(T).any(axis=1)
            T = T[~bad_rows, :]; S = S[~bad_rows, :]; depth_grid = depth_grid[~bad_rows]

            depth_grid = np.abs(depth_grid.astype(float))
            order = np.argsort(depth_grid)
            depth_grid = depth_grid[order]; T = T[order, :]; S = S[order, :]

            n_time = T.shape[1]
            if n_time < TSF_K + TSF_H:
                continue

            try:
                time_dts = [_datenum2datetime(dn) for dn in time_grid]
            except Exception:
                continue

            thermo_depths = _compute_thermo_depths(T, depth_grid)

            # manifest 划分为文件级，整个文件全部使用
            t_start, t_end = 0, n_time

            n_before = len(self.samples)
            # 关键：保证整段窗口 [start, start+K+H) 完全落在当前 split 内，
            # 避免 train/val/test 之间出现时间泄漏。
            window_len = TSF_K + TSF_H
            start_min = t_start
            start_max_exclusive = min(t_end, n_time) - window_len + 1
            if start_max_exclusive <= start_min:
                continue

            for start in range(start_min, start_max_exclusive, CTD_TSF_STRIDE):
                mid_dt  = time_dts[start + TSF_K // 2]
                sid     = _season_id(mid_dt)
                doy     = mid_dt.timetuple().tm_yday
                doy_sin = float(np.sin(2 * np.pi * doy / 365))
                doy_cos = float(np.cos(2 * np.pi * doy / 365))
                target_depths = [
                    _to_model_depth(thermo_depths[t])
                    for t in range(start + TSF_K, start + TSF_K + TSF_H)
                ]
                for _ in range(CTD_TSF_N_AUGMENT):
                    input_depths, input_features, input_sal_features = [], [], []

                    # 模拟 AUV K 步轨迹，累积 (depth, T, S) 历史观测
                    traj_d, traj_T, traj_S = [], [], []
                    for t in range(start, start + TSF_K):
                        if rng.random() < CTD_TSF_NEAR_THERMO_PROB:
                            auv_d = float(thermo_depths[t]) + float(rng.normal(0, CTD_TSF_AUV_NOISE_STD))
                        else:
                            auv_d = float(rng.uniform(abs(DEPTH_MAX), abs(DEPTH_MIN)))
                        auv_d = float(np.clip(auv_d, abs(DEPTH_MAX), abs(DEPTH_MIN)))
                        traj_d.append(auv_d)

                        # 在当前时间步 t 的剖面上插值得到该深度的 T/S
                        f_T = interp1d(depth_grid, T[:, t], kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
                        f_S = interp1d(depth_grid, S[:, t], kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
                        traj_T.append(float(f_T(auv_d)))
                        traj_S.append(float(f_S(auv_d)))

                        input_depths.append(_to_model_depth(auv_d))

                        # PO：用截至当前步的累积轨迹按深度分 bin 取最新值
                        d_arr = np.array(traj_d)
                        T_arr = np.array(traj_T)
                        S_arr = np.array(traj_S)
                        d_lo, d_hi = d_arr.min(), d_arr.max()
                        w = SAMPLE_WINDOW_SIZE
                        if d_hi - d_lo < STEP_SIZE:
                            po_T = [T_arr[-1]] * w
                            po_S = [S_arr[-1]] * w
                        else:
                            bin_edges = np.linspace(d_lo, d_hi, w + 1)
                            po_T = [T_arr[-1]] * w
                            po_S = [S_arr[-1]] * w
                            for i in range(w):
                                mask = np.where(
                                    (d_arr >= bin_edges[i]) & (d_arr < bin_edges[i + 1])
                                )[0]
                                if len(mask) > 0:
                                    po_T[i] = float(T_arr[mask[-1]])
                                    po_S[i] = float(S_arr[mask[-1]])
                        input_features.append(po_T)
                        input_sal_features.append(po_S)
                    self.samples.append({
                        'episode_id':         file_idx * 1_000_000 + start,
                        'season_id':          sid,
                        'doy':                doy,
                        'doy_sin':            doy_sin,
                        'doy_cos':            doy_cos,
                        'input_depths':       input_depths,
                        'input_features':     input_features,
                        'input_sal_features': input_sal_features,
                        'target_depths':      target_depths,
                    })

            n_file = len(self.samples) - n_before
            if n_file and self._verbose:
                print(f"  {fname}: {n_time} steps  samples={n_file}")

        if cache_file is not None:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
            if verbose:
                print(f"[CTDTSFDataset] 已缓存至 {cache_file}  ({len(self.samples)} 样本)")

    def __len__(self):   return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


if __name__ == '__main__':
    ctd_train = CTDTSFDataset(split='train')
    ctd_val   = CTDTSFDataset(split='validation')
    print(f"[CTD-TSF]  train: {len(ctd_train)}  val: {len(ctd_val)}")
    if ctd_train:
        print(ctd_train[0].keys())
