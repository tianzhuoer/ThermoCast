"""
Chronos-2 微调训练入口：在 CTD 数据集上对 Chronos-2 进行监督微调。

启动命令
--------
单 GPU（指定数据量）:
  python train_chronos2.py --samples 36000 --gpu 1
多 GPU:
  accelerate launch --num_processes=<N> train_chronos2.py --samples 36000

参数说明
--------
--samples  本次训练使用的样本数（默认全量）
--gpu      指定 CUDA 设备编号，如 0、1（多卡时忽略）

说明
----
输入风格与 ThermoForecasterReprogram 对齐：
  depths(B,K,1) + T_profs(B,K,N) + S_profs(B,K,N) + context(B,3)
每个模态经 per-modality Linear 压缩为标量序列，4 条序列用 group_ids 送入 backbone。
"""

import argparse as _ap
import os

os.environ["HF_HUB_OFFLINE"] = "1"  # 必须在所有 import 之前设置，阻止 chronos 启动时联网

# CUDA_VISIBLE_DEVICES 必须在 torch import 之前设置，否则无效
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--gpu", default=None)
_gpu_arg, _ = _p.parse_known_args()
if _gpu_arg.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg.gpu

import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import configs.chronos2_tsf as cfg
from utils import apply_gpu_arg, apply_samples_arg
from models.chronos2_wrapper import load_chronos2
from trainers.tsf_trainer import TSFTrainer, build_datasets

# ── 目标深度归一化 ─────────────────────────────────────────────────────────────
_D_RANGE = cfg.TSF_DEPTH_MAX - cfg.TSF_DEPTH_MIN

def _norm_target(z):
    return (z - cfg.TSF_DEPTH_MIN) / _D_RANGE * 2.0 - 1.0


# ── collate_fn ────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    与 train_reprogram_TSF.py 风格一致：返回原始张量，不做特征聚合。

    返回：
      depths:   (B, K, 1)   AUV 深度（米，负值）
      T_profs:  (B, K, N)   温度局部剖面
      S_profs:  (B, K, N)   盐度局部剖面
      context:  (B, 3)      [season_norm, doy_sin, doy_cos]
      targets:  (B, H)      归一化目标深度 [-1, 1]
    """
    depths_list, T_list, S_list, ctx_list, tgt_list = [], [], [], [], []

    for ex in batch:
        input_depths = list(ex["input_depths"])
        input_feats  = [list(f) for f in ex["input_features"]]
        input_sal    = [list(f) for f in ex.get("input_sal_features", [])]
        has_sal      = len(input_sal) == len(input_feats)

        K = len(input_feats)
        N = len(input_feats[0])

        depths_t = torch.tensor([[float(d)] for d in input_depths],
                                dtype=torch.float32)              # (K, 1)
        T_t = torch.tensor(input_feats, dtype=torch.float32)     # (K, N)
        S_t = (torch.nan_to_num(torch.tensor(input_sal, dtype=torch.float32), nan=0.0)
               if has_sal
               else torch.zeros(K, N, dtype=torch.float32))      # (K, N)

        season_norm = float(ex["season_id"]) / 3.0
        ctx_t = torch.tensor(
            [season_norm, float(ex["doy_sin"]), float(ex["doy_cos"])],
            dtype=torch.float32,
        )                                                          # (3,)

        tgt_t = torch.tensor(
            [_norm_target(float(z)) for z in ex["target_depths"]],
            dtype=torch.float32,
        )                                                          # (H,)

        depths_list.append(depths_t)
        T_list.append(T_t)
        S_list.append(S_t)
        ctx_list.append(ctx_t)
        tgt_list.append(tgt_t)

    return (
        torch.stack(depths_list),  # (B, K, 1)
        torch.stack(T_list),       # (B, K, N)
        torch.stack(S_list),       # (B, K, N)
        torch.stack(ctx_list),     # (B, 3)
        torch.stack(tgt_list),     # (B, H)
    )


# ── forward_fn ────────────────────────────────────────────────────────────────

def forward_fn(model, batch):
    depths, T_profs, S_profs, context, targets = batch
    preds = model(depths, T_profs, S_profs, context)
    loss  = F.huber_loss(preds, targets, delta=cfg.HUBER_DELTA)
    return loss, preds, targets


# ── save_fn ───────────────────────────────────────────────────────────────────

def save_fn(model, run_id, epoch, val_mae, ckpt_dir, is_best):
    target_dir = cfg.TSF_BEST_DIR if is_best else os.path.join(ckpt_dir, "latest")
    if not is_best and os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    torch.save(model.base.state_dict(),
               os.path.join(target_dir, "chronos2_backbone.pt"))
    for name in ("T_proj", "S_proj", "projection_head"):
        torch.save(getattr(model, name).state_dict(),
                   os.path.join(target_dir, f"{name}.pt"))

    if is_best:
        print(f"  → 已保存最优模型至 {target_dir}")
    else:
        print(f"  → 检查点已保存: {target_dir}  (ep{epoch} mae={val_mae:.2f}m)")


# ── 主流程 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_gpu_arg(cfg.CUDA_DEVICE_LLM)
    apply_samples_arg(cfg)
    accelerator = Accelerator(gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS)

    random.seed(cfg.MANUAL_SEED)
    np.random.seed(cfg.MANUAL_SEED)
    torch.manual_seed(cfg.MANUAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.MANUAL_SEED)

    train_dataset, val_dataset, baselines = build_datasets(cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    model = load_chronos2(cfg)
    wd_head = getattr(cfg, 'WEIGHT_DECAY_HEAD', cfg.WEIGHT_DECAY)

    head_params = (list(model.T_proj.parameters()) +
                   list(model.S_proj.parameters()) +
                   list(model.projection_head.parameters()))
    head_groups = [{"params": head_params, "lr": cfg.LR_FT_HEAD, "weight_decay": wd_head}]

    if cfg.FREEZE_BACKBONE:
        param_groups = head_groups
    else:
        param_groups = [
            {"params": list(model.base.parameters()), "lr": cfg.LR_FT_BACKBONE, "weight_decay": cfg.WEIGHT_DECAY},
        ] + head_groups
    optimizer = AdamW(param_groups)

    total_steps  = len(train_loader) * cfg.EPOCHS // cfg.GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    trainer = TSFTrainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        forward_fn=forward_fn,
        save_fn=save_fn,
        baseline_metrics=baselines,
        swanlab_extra_cfg={
            "model":                cfg.CHRONOS2_MODEL_ID,
            "freeze_backbone":      cfg.FREEZE_BACKBONE,
            "lr_backbone":          cfg.LR_FT_BACKBONE,
            "lr_head":              cfg.LR_FT_HEAD,
            "epochs":               cfg.EPOCHS,
            "batch_size":           cfg.BATCH_SIZE,
            "tsf_K":                cfg.TSF_K,
            "tsf_H":                cfg.TSF_H,
            "seed":                 cfg.MANUAL_SEED,
            "warmup_steps":         warmup_steps,
            "total_steps":          total_steps,
            "grad_accum_steps":     cfg.GRAD_ACCUM_STEPS,
            "effective_batch_size": cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS,
            "n_modalities":         3,
            "weight_decay":         cfg.WEIGHT_DECAY,
            "weight_decay_head":    wd_head,
            "huber_delta":          cfg.HUBER_DELTA,
            "es_patience":          cfg.ES_PATIENCE,
            "es_min_epochs":        cfg.ES_MIN_EPOCHS,
        },
    )
    trainer.train()
