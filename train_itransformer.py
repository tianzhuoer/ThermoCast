"""
iTransformer 基线训练入口：在 CTD 数据集上从零训练 iTransformer 温跃层预测模型。

iTransformer（ICLR 2024）将每个特征维度视为 token，在特征间做 Self-Attention，
捕获跨特征相关性，FFN 在 d_model 空间内提取各特征的时间模式。

输入与 ThermoForecasterReprogram 对齐：四路原始剖面（depths/T/S/context），
归一化在模型内部完成，collate_fn 与 train_reprogram_TSF.py 保持一致。

启动命令
--------
单 GPU（指定数据量）:
  python train_itransformer.py --samples 16800 --gpu 1
多 GPU:
  accelerate launch --num_processes=<N> train_itransformer.py --samples 16800

参数说明
--------
--samples  本次训练使用的样本数（默认全量）
--gpu      指定 CUDA 设备编号，如 0、1（多卡时忽略）
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

import configs.itransformer_tsf as cfg
from utils import apply_gpu_arg, apply_samples_arg
from models.seq_baselines import load_itransformer
from trainers.tsf_trainer import TSFTrainer, build_datasets

_D_RANGE = cfg.TSF_DEPTH_MAX - cfg.TSF_DEPTH_MIN

def _norm_target(z):
    return (z - cfg.TSF_DEPTH_MIN) / _D_RANGE * 2.0 - 1.0


def collate_fn(batch, is_train=False):
    depths_list, T_list, S_list, ctx_list, tgt_list = [], [], [], [], []

    for ex in batch:
        input_depths = list(ex["input_depths"])
        input_feats  = [list(f) for f in ex["input_features"]]
        input_sal    = [list(f) for f in ex.get("input_sal_features", [])]
        has_sal      = len(input_sal) == len(input_feats)

        if is_train:
            input_depths = [d + random.gauss(0, cfg.DEPTH_NOISE_STD) for d in input_depths]
            input_feats  = [[v + random.gauss(0, cfg.FEAT_NOISE_STD) for v in row]
                            for row in input_feats]
            if has_sal:
                input_sal = [[v + random.gauss(0, cfg.FEAT_NOISE_STD) for v in row]
                             for row in input_sal]

        K = len(input_feats)
        N = len(input_feats[0])

        depths_t = torch.tensor([[float(d)] for d in input_depths], dtype=torch.float32)
        T_t      = torch.tensor(input_feats, dtype=torch.float32)
        S_t      = (torch.nan_to_num(torch.tensor(input_sal, dtype=torch.float32), nan=0.0)
                    if has_sal else torch.zeros(K, N, dtype=torch.float32))

        season_norm = float(ex["season_id"]) / 3.0
        ctx_t = torch.tensor(
            [season_norm, float(ex["doy_sin"]), float(ex["doy_cos"])],
            dtype=torch.float32,
        )
        tgt_t = torch.tensor(
            [_norm_target(float(z)) for z in ex["target_depths"]],
            dtype=torch.float32,
        )

        depths_list.append(depths_t)
        T_list.append(T_t)
        S_list.append(S_t)
        ctx_list.append(ctx_t)
        tgt_list.append(tgt_t)

    return (
        torch.stack(depths_list),
        torch.stack(T_list),
        torch.stack(S_list),
        torch.stack(ctx_list),
        torch.stack(tgt_list),
    )


def collate_train(batch): return collate_fn(batch, is_train=True)
def collate_val(batch):   return collate_fn(batch, is_train=False)


def forward_fn(model, batch):
    depths, T_profs, S_profs, context, targets = batch
    preds = model(depths, T_profs, S_profs, context)
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

    model     = load_itransformer(cfg)
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
            "model":           "iTransformer",
            "freeze_backbone": False,
            "lr_backbone":     cfg.LR,
            "lr_head":         cfg.LR,
            "d_model":         cfg.IT_D_MODEL,
            "n_heads":         cfg.IT_N_HEADS,
            "num_layers":      cfg.IT_NUM_LAYERS,
            "ffn_dim":         cfg.IT_FFN_DIM,
            "dropout":         cfg.IT_DROPOUT,
            "epochs":          cfg.EPOCHS,
            "batch_size":      cfg.BATCH_SIZE,
            "tsf_K":           cfg.TSF_K,
            "tsf_H":           cfg.TSF_H,
            "n_input_features": 1 + cfg.SAMPLE_WINDOW_SIZE * 2 + 3,  # depth+T+S+ctx
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
