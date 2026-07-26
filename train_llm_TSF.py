"""
LLM-TSF 训练入口：Qwen3 + LoRA 微调温跃层中心深度时序预测。

启动命令
--------
多 GPU:  accelerate launch --num_processes=<N> train_llm_TSF.py
单 GPU:  python train_llm_TSF.py
"""

import os
import random
import shutil
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import configs.llm_tsf as cfg
from models.llm_backbone import ThermoForecaster, load_backbone
from utils import apply_gpu_arg
from trainers.tsf_trainer import TSFTrainer, build_datasets

# ── 深度归一化 ─────────────────────────────────────────────────────────────────
_D_RANGE = cfg.TSF_DEPTH_MAX - cfg.TSF_DEPTH_MIN

def _norm(z):
    return (z - cfg.TSF_DEPTH_MIN) / _D_RANGE * 2.0 - 1.0

def _add_noise(lst, std):
    return [v + random.gauss(0, std) for v in lst]

def _add_noise_2d(lst2d, std):
    return [[v + random.gauss(0, std) for v in row] for row in lst2d]

def _step_stats(features):
    if not features or len(features) < 2:
        return 0.0, 0.0
    vals = [float(v) for v in features]
    max_grad = max(abs(vals[i+1] - vals[i]) for i in range(len(vals) - 1))
    return max_grad, sum(vals) / len(vals)


def _format_full_obs(d, mg_t, avg_t, mg_s=None, avg_s=None):
    if mg_s is None or avg_s is None:
        return f"{d:.1f} {mg_t:.3f} {avg_t:.1f}"
    return f"{d:.1f} {mg_t:.3f} {avg_t:.1f} {mg_s:.3f} {avg_s:.1f}"


def _q(value, scale):
    value = float(value)
    if not np.isfinite(value):
        return 0
    return round(value * scale)


def _format_compact_obs(d, mg_t, avg_t, mg_s=None, avg_s=None):
    vals = [
        _q(d, 10),
        _q(mg_t, 1000),
        _q(avg_t, 10),
    ]
    if mg_s is not None and avg_s is not None:
        vals.extend([_q(mg_s, 1000), _q(avg_s, 10)])
    return " ".join(str(v) for v in vals)


def _format_compact_summary(rows):
    if not rows:
        return "E 0 0 0 0 0"

    arr = np.asarray(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    depth = arr[:, 0]
    t_grad = arr[:, 1]
    avg_t = arr[:, 2]
    vals = [
        _q(depth[0], 10),
        _q(depth[-1], 10),
        _q(depth.mean(), 10),
        _q(t_grad.mean(), 1000),
        _q(t_grad.max(), 1000),
        _q(avg_t.mean(), 10),
    ]
    if arr.shape[1] >= 5:
        s_grad = arr[:, 3]
        avg_s = arr[:, 4]
        vals.extend([
            _q(s_grad.mean(), 1000),
            _q(s_grad.max(), 1000),
            _q(avg_s.mean(), 10),
        ])
    return "E " + " ".join(str(v) for v in vals)


def _build_full_text(season_id, season_name, typical_dep, doy, doy_sin, doy_cos,
                     stat_rows, has_sal):
    obs_lines = [
        _format_full_obs(*row) if has_sal else _format_full_obs(*row[:3])
        for row in stat_rows
    ]
    col_header = "depth T_grad avg_T S_grad avg_S" if has_sal else "depth T_grad avg_T"
    system_prompt = cfg.PROMPT_TSF.format(season=season_name, H=cfg.TSF_H)
    user_content = (
        f"{season_name} DOY:{doy} sin:{doy_sin:.3f} cos:{doy_cos:.3f} typical:{typical_dep}\n"
        f"{col_header}\n"
        + "\n".join(obs_lines)
        + f"\nPredict next {cfg.TSF_H} depths."
    )
    return (
        f"<s><|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _build_compact_text(season_id, doy, stat_rows, has_sal):
    recent_steps = min(getattr(cfg, "COMPACT_RECENT_STEPS", 10), len(stat_rows))
    early_rows = stat_rows[:-recent_steps] if recent_steps else stat_rows
    recent_rows = stat_rows[-recent_steps:] if recent_steps else []

    use_sal = has_sal and getattr(cfg, "COMPACT_KEEP_SALINITY", True)
    recent_lines = [
        _format_compact_obs(*row) if use_sal else _format_compact_obs(*row[:3])
        for row in recent_rows
    ]
    summary_rows = [row if use_sal else row[:3] for row in early_rows]
    cols = "d tg t sg s" if use_sal else "d tg t"
    user_content = (
        f"S {season_id} D {doy} H {cfg.TSF_H}\n"
        f"C {cols}\n"
        f"{_format_compact_summary(summary_rows)}\n"
        "R\n"
        + "\n".join(recent_lines)
    )
    return (
        f"<s><|im_start|>system\nTSF<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── collate_fn ────────────────────────────────────────────────────────────────

def collate_fn(batch, tokenizer, is_train=False):
    """将 CTDTSFDataset 样本转为 (input_ids, attention_mask, targets_norm) 三元组。"""
    input_ids_list, mask_list, targets_list = [], [], []

    for ex in batch:
        season_id   = int(ex['season_id'])
        season_name = cfg.SEASON_NAMES[season_id]
        typical_dep = cfg.SEASON_TYPICAL_DEPTHS[season_id]
        doy         = int(ex['doy'])
        doy_sin     = float(ex['doy_sin'])
        doy_cos     = float(ex['doy_cos'])

        depths       = list(ex['input_depths'])
        features     = [list(f) for f in ex['input_features']]
        sal_features = [list(f) for f in ex.get('input_sal_features', [])]
        has_sal      = len(sal_features) == len(features)

        if is_train:
            depths   = _add_noise(depths, cfg.DEPTH_NOISE_STD)
            features = _add_noise_2d(features, cfg.FEAT_NOISE_STD)
            if has_sal:
                sal_features = _add_noise_2d(sal_features, cfg.FEAT_NOISE_STD)

        stat_rows = []
        for k, (d, feat) in enumerate(zip(depths, features)):
            mg_t, avg_t = _step_stats(feat)
            if has_sal:
                mg_s, avg_s = _step_stats(sal_features[k])
                stat_rows.append((d, mg_t, avg_t, mg_s, avg_s))
            else:
                stat_rows.append((d, mg_t, avg_t))

        if getattr(cfg, "TSF_INPUT_FORMAT", "full") == "compact":
            full_text = _build_compact_text(season_id, doy, stat_rows, has_sal)
        else:
            full_text = _build_full_text(
                season_id, season_name, typical_dep, doy, doy_sin, doy_cos,
                stat_rows, has_sal,
            )

        enc = tokenizer(
            full_text,
            add_special_tokens=False,
            max_length=cfg.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids_list.append(enc['input_ids'])
        mask_list.append(enc['attention_mask'])
        targets_list.append([_norm(float(z)) for z in ex['target_depths']])

    return (
        torch.cat(input_ids_list, dim=0),
        torch.cat(mask_list,      dim=0),
        torch.tensor(targets_list, dtype=torch.float32),
    )


# ── forward_fn ────────────────────────────────────────────────────────────────

def forward_fn(model, batch):
    input_ids, attention_mask, targets = batch
    preds = model(input_ids, attention_mask)
    loss  = F.huber_loss(preds, targets, delta=cfg.HUBER_DELTA)
    return loss, preds, targets


# ── save_fn ────────────────────────────────────────────────────────────────────

def save_fn(model, run_id, epoch, val_mae, ckpt_dir, is_best):
    if is_best:
        model.backbone.save_pretrained(cfg.TSF_LORA_BEST)
        torch.save(model.regressor.state_dict(), cfg.TSF_HEAD_BEST)
        print(f"  → 已保存最优模型至 {cfg.TSF_LORA_BEST}")
    else:
        lora_dir  = os.path.join(ckpt_dir, "lora_latest")
        head_path = os.path.join(ckpt_dir, "head_latest.pt")
        if os.path.exists(lora_dir):
            shutil.rmtree(lora_dir)
        os.makedirs(lora_dir, exist_ok=True)
        model.backbone.save_pretrained(lora_dir)
        torch.save(model.regressor.state_dict(), head_path)
        print(f"  → 检查点已保存: {lora_dir}  (ep{epoch} mae={val_mae:.2f}m)")


# ── 主流程 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    accelerator = Accelerator(gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS)

    random.seed(cfg.MANUAL_SEED)
    np.random.seed(cfg.MANUAL_SEED)
    torch.manual_seed(cfg.MANUAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.MANUAL_SEED)

    apply_gpu_arg(cfg.CUDA_DEVICE_LLM)

    train_dataset, val_dataset, baselines = build_datasets(cfg)

    backbone, tokenizer = load_backbone(cfg)
    forecaster = ThermoForecaster(backbone, H=cfg.TSF_H)

    _collate_train = partial(collate_fn, tokenizer=tokenizer, is_train=True)
    _collate_val   = partial(collate_fn, tokenizer=tokenizer, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  collate_fn=_collate_train)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=_collate_val)

    lora_params = [p for p in forecaster.backbone.parameters() if p.requires_grad]
    head_params = list(forecaster.regressor.parameters())
    optimizer = AdamW([
        {"params": lora_params, "lr": cfg.LR_LORA, "weight_decay": 0.0},
        {"params": head_params, "lr": cfg.LR_HEAD,  "weight_decay": cfg.WEIGHT_DECAY},
    ])
    total_steps  = len(train_loader) * cfg.EPOCHS // cfg.GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    trainer = TSFTrainer(
        cfg=cfg,
        model=forecaster,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        forward_fn=forward_fn,
        save_fn=save_fn,
        baseline_metrics=baselines,
        swanlab_extra_cfg={
            "lr_lora": cfg.LR_LORA, "lr_head": cfg.LR_HEAD,
            "epochs": cfg.EPOCHS, "batch_size": cfg.BATCH_SIZE,
            "max_length": cfg.MAX_LENGTH, "seed": cfg.MANUAL_SEED,
            "input_format": getattr(cfg, "TSF_INPUT_FORMAT", "full"),
            "compact_recent_steps": getattr(cfg, "COMPACT_RECENT_STEPS", None),
            "compact_keep_salinity": getattr(cfg, "COMPACT_KEEP_SALINITY", None),
            "lora_r": cfg.LORA_R, "lora_alpha": cfg.LORA_ALPHA, "lora_dropout": cfg.LORA_DROPOUT,
            "tsf_K": cfg.TSF_K, "tsf_H": cfg.TSF_H,
            "warmup_steps": warmup_steps, "total_steps": total_steps,
            "grad_accum_steps":     cfg.GRAD_ACCUM_STEPS,
            "effective_batch_size": cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS,
            "weight_decay":         cfg.WEIGHT_DECAY,
            "huber_delta":          cfg.HUBER_DELTA,
            "es_patience":          cfg.ES_PATIENCE,
            "es_min_epochs":        cfg.ES_MIN_EPOCHS,
        },
    )
    trainer.train()
