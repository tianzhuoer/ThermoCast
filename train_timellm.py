"""
Time-LLM（ICLR 2024）baseline 训练入口 —— 架构对照。

与主方法 Reprogram-TSF 同 backbone（Qwen3.5-2B）+ 同 LoRA 策略，仅架构换成官方
Time-LLM 那套（时间轴 patch + text prototypes mapping + Prompt-as-Prefix）。
输入与其它 baseline 对齐：四路原始剖面（depths/T/S/context），内部归一化后在变量维展平。

启动命令
--------
单 GPU:
  python train_timellm.py --samples 16800 --gpu 1
多 GPU:
  accelerate launch --num_processes=<N> train_timellm.py --samples 16800

参数说明
--------
--samples  本次训练使用的样本数（默认全量）
--gpu      指定 CUDA 设备编号，如 0、1（多卡时忽略）
"""

import argparse as _ap
import os

# CUDA_VISIBLE_DEVICES 必须在 torch import 之前设置
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

import configs.timellm_tsf as cfg
from utils import apply_gpu_arg, apply_samples_arg
from models.timellm_wrapper import load_timellm
from trainers.tsf_trainer import TSFTrainer, build_datasets

_D_RANGE = cfg.TSF_DEPTH_MAX - cfg.TSF_DEPTH_MIN


def _norm_target(z):
    return (z - cfg.TSF_DEPTH_MIN) / _D_RANGE * 2.0 - 1.0


# ── collate_fn（与 train_itransformer 一致：四路剖面 + 3 维 context）──────────────

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
    if not torch.isfinite(loss):
        print(f"[WARN] non-finite loss={loss.item():.4f}, skipping batch")
        loss = torch.zeros(1, device=loss.device, requires_grad=True).squeeze()
    return loss, preds, targets


def save_fn(model, run_id, epoch, val_mae, ckpt_dir, is_best):
    """LoRA backbone 与非 backbone 参数分开保存（与 reprogram 一致）。"""
    non_backbone = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("backbone.")
    }
    if is_best:
        if os.path.exists(cfg.TSF_LORA_BEST):
            shutil.rmtree(cfg.TSF_LORA_BEST)
        model.backbone.save_pretrained(cfg.TSF_LORA_BEST)
        torch.save({"timellm": non_backbone}, cfg.TSF_HEAD_BEST)
        print(f"  → 最优模型已保存至 {cfg.TSF_HEAD_BEST}")
    else:
        head_path = os.path.join(ckpt_dir, "head_latest.pt")
        lora_dir  = os.path.join(ckpt_dir, "lora_latest")
        if os.path.exists(lora_dir):
            shutil.rmtree(lora_dir)
        os.makedirs(lora_dir, exist_ok=True)
        model.backbone.save_pretrained(lora_dir)
        torch.save({"timellm": non_backbone}, head_path)
        print(f"  → 检查点已保存: {ckpt_dir}  (ep{epoch} mae={val_mae:.2f}m)")


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

    model = load_timellm(cfg)

    # ── 分组学习率：LoRA 较低；time-llm 架构组件（patch/reprogram/mapping/out_proj）
    #    + head 较高（与 reprogram 一致的分组思路）──────────────────────────────
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    arch_params = (
        list(model.patch_embedding.parameters())
        + list(model.reprogramming_layer.parameters())
        + [model.prototypes]
        + list(model.out_proj.parameters())
    )
    head_params = list(model.head.parameters())

    param_groups = [
        {"params": lora_params, "lr": cfg.LR_LORA,      "weight_decay": 0.0,             "name": "lora"},
        {"params": arch_params, "lr": cfg.LR_REPROGRAM, "weight_decay": cfg.WEIGHT_DECAY, "name": "arch"},
        {"params": head_params, "lr": cfg.LR_HEAD,      "weight_decay": cfg.WEIGHT_DECAY, "name": "head"},
    ]
    optimizer = AdamW(param_groups)

    lora_n = sum(p.numel() for p in lora_params)
    arch_n = sum(p.numel() for p in arch_params + head_params)
    print(f"[check] LoRA trainable: {lora_n:,}  (应 >0 才说明 LoRA 生效)")
    print(f"[check] arch+head trainable: {arch_n:,}  "
          f"(lr_arch={cfg.LR_REPROGRAM}, lr_head={cfg.LR_HEAD}, wd={cfg.WEIGHT_DECAY})")

    total_steps  = len(train_loader) * cfg.EPOCHS // cfg.GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * getattr(cfg, "WARMUP_RATIO", 0.05))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
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
            "model":           "Time-LLM",
            "freeze_backbone": False,
            "backbone":        "Qwen3.5-2B",
            "lr_lora":         cfg.LR_LORA,
            "lr_arch":         cfg.LR_REPROGRAM,
            "lr_head":         cfg.LR_HEAD,
            "lora_r":          cfg.LORA_R,
            "lora_alpha":      cfg.LORA_ALPHA,
            "patch_len":       cfg.TIMELLM_PATCH_LEN,
            "stride":          cfg.TIMELLM_STRIDE,
            "d_model":         cfg.TIMELLM_D_MODEL,
            "n_heads":         cfg.TIMELLM_N_HEADS,
            "n_prototypes":    cfg.TIMELLM_N_PROTOTYPES,
            "d_ff":            cfg.TIMELLM_D_FF,
            "epochs":          cfg.EPOCHS,
            "batch_size":      cfg.BATCH_SIZE,
            "tsf_K":           cfg.TSF_K,
            "tsf_H":           cfg.TSF_H,
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
