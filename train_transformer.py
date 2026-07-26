"""
Transformer 基线训练入口：在 CTD 数据集上从零训练 Transformer Encoder 温跃层预测模型。

输入使用 34 维原始剖面（与 LSTM 一致），d_model=32 小容量模型作为弱 baseline。

启动命令
--------
单 GPU:  python train_transformer.py
多 GPU:  accelerate launch --num_processes=<N> train_transformer.py
"""

import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import configs.transformer_tsf as cfg
from utils import apply_gpu_arg, apply_samples_arg
from models.seq_baselines import load_transformer
from trainers.tsf_trainer import TSFTrainer, build_datasets

_D_RANGE = cfg.TSF_DEPTH_MAX - cfg.TSF_DEPTH_MIN

def _norm(z):
    return (z - cfg.TSF_DEPTH_MIN) / _D_RANGE * 2.0 - 1.0


def _make_raw_feature(ex, is_train: bool) -> torch.Tensor:
    input_depths = list(ex["input_depths"])
    input_feats  = [list(f) for f in ex["input_features"]]
    input_sal    = [list(f) for f in ex.get("input_sal_features", [])]
    K = len(input_feats)
    N = len(input_feats[0])
    has_sal = len(input_sal) == K

    if is_train:
        input_depths = [d + random.gauss(0, cfg.DEPTH_NOISE_STD) for d in input_depths]
        input_feats  = [[v + random.gauss(0, cfg.FEAT_NOISE_STD) for v in row] for row in input_feats]
        if has_sal:
            input_sal = [[v + random.gauss(0, cfg.FEAT_NOISE_STD) for v in row] for row in input_sal]

    d_range    = cfg.TSF_DEPTH_MAX - cfg.TSF_DEPTH_MIN
    depth_norm = [(d - cfg.TSF_DEPTH_MIN) / d_range * 2.0 - 1.0 for d in input_depths]

    T_arr  = np.array(input_feats, dtype=np.float32)
    T_norm = (T_arr - T_arr.mean(1, keepdims=True)) / (T_arr.std(1, keepdims=True) + 1e-6)

    if has_sal:
        S_arr  = np.array(input_sal, dtype=np.float32)
        S_norm = (S_arr - S_arr.mean(1, keepdims=True)) / (S_arr.std(1, keepdims=True) + 1e-6)
    else:
        S_norm = np.zeros((K, N), dtype=np.float32)

    depth_col = np.array(depth_norm, dtype=np.float32).reshape(K, 1)
    feat = np.concatenate([depth_col, T_norm, S_norm], axis=1)  # 不含季节/doy，31维
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(feat, dtype=torch.float32)


def collate_fn(batch, is_train=False):
    feat_list    = [_make_raw_feature(ex, is_train) for ex in batch]
    targets_list = [[_norm(float(z)) for z in ex['target_depths']] for ex in batch]
    return torch.stack(feat_list), torch.tensor(targets_list, dtype=torch.float32)


def collate_train(batch): return collate_fn(batch, is_train=True)
def collate_val(batch):   return collate_fn(batch, is_train=False)


def forward_fn(model, batch):
    features, targets = batch
    preds = model(features)
    loss  = F.huber_loss(preds, targets, delta=cfg.HUBER_DELTA)
    return loss, preds, targets


def save_fn(model, run_id, epoch, val_mae, ckpt_dir, is_best):
    if is_best:
        os.makedirs(cfg.TSF_BEST_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(cfg.TSF_BEST_DIR, "model.pt"))
        print(f"  → 已保存最优模型至 {cfg.TSF_BEST_DIR}")
    else:
        save_dir = os.path.join(ckpt_dir, "latest")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        print(f"  → 检查点已保存: {save_dir}  (ep{epoch} mae={val_mae:.2f}m)")


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
                              shuffle=True,  collate_fn=collate_train, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_val,   num_workers=0)

    model     = load_transformer(cfg)
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

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
            "model":           "Transformer",
            "freeze_backbone": False,
            "lr_backbone":     cfg.LR,
            "lr_head":         cfg.LR,
            "d_model":         cfg.TF_D_MODEL,
            "n_heads":         cfg.TF_N_HEADS,
            "num_layers":      cfg.TF_NUM_LAYERS,
            "ffn_dim":         cfg.TF_FFN_DIM,
            "dropout":         cfg.TF_DROPOUT,
            "epochs":     cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "tsf_K":      cfg.TSF_K,
            "tsf_H":      cfg.TSF_H,
            "n_input_features": cfg.RAW_INPUT_SIZE,
            "seed":                 cfg.MANUAL_SEED,
            "warmup_steps":         warmup_steps,
            "total_steps":          total_steps,
            "grad_accum_steps":     cfg.GRAD_ACCUM_STEPS,
            "effective_batch_size": cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS,
            "weight_decay":         cfg.WEIGHT_DECAY,
            "huber_delta":          cfg.HUBER_DELTA,
            "es_patience":          cfg.ES_PATIENCE,
            "es_min_epochs":        cfg.ES_MIN_EPOCHS,
        },
    )
    trainer.train()
