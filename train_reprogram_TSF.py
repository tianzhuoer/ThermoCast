"""
Reprogram-TSF 训练入口：Qwen3.5-2B + LoRA + ReprogrammingLayer。

每个时间步按数据类型（depth/T-profile/S-profile）各生成一个 embedding token，
通过 ReprogrammingLayer 对齐到 LLM embedding 空间，与文本 prompt 拼接后送入 Qwen backbone。

token 数：
  静态 prompt:  L_prompt + 3×K = ~36 + 3×32 = ~132（季节定长模板，[THERMO]→ctx_token）
  动态 prompt:  L_prompt + 3×K = ~101 + 96 = ~197（窗口统计量文本，prompt 占比 ~51%）

启动命令
--------
单 GPU（指定模式和数据量；head 与 lora 除 backbone 是否训练外，超参完全一致）:
  python train_reprogram_TSF.py --mode lora  --samples 36000 --gpu 1
  python train_reprogram_TSF.py --mode head  --samples 36000 --gpu 1
多 GPU:
  accelerate launch --num_processes=<N> train_reprogram_TSF.py --mode lora --samples 36000
  accelerate launch --num_processes=<N> train_reprogram_TSF.py --mode head --samples 36000

消融实验（w/o reprogram，用 Linear 替换 ReprogrammingLayer）:
  python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 1 --no-reprogram
  python train_reprogram_TSF.py --mode head --samples 16800 --gpu 1 --no-reprogram

诊断消融（验证 LLM backbone 是否真的参与预测）:
  python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 1 --no-backbone
  python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 1 --random-init

动态 prompt 实验（把窗口统计量文本化，激活预训练语义；三条配套对照）:
  # 1) 动态 prompt + 真预训练（主角）
  python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 1 --dynamic-prompt
  # 2) 动态 prompt + 随机权重（验证激活的是否为"预训练语义"）
  python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 1 --dynamic-prompt --random-init
  # 3) 动态 prompt + 冻结 LLM（验证微调是否开始有意义）
  python train_reprogram_TSF.py --mode head --samples 16800 --gpu 1 --dynamic-prompt
  判读：(1) 明显优于 (2) → 被激活的是预训练语义；与无 prompt 基线比 (3)vs微调出现差距 → 知识被激活

参数说明
--------
--mode          训练模式：lora（LoRA+reprogram+head）或 head（冻结 LLM，仅 reprogram+head；超参与 lora 一致）
--samples       本次训练使用的样本数（默认全量）
--gpu           指定 CUDA 设备编号，如 0、1（多卡时忽略）
--no-reprogram  消融开关：用 Linear(d_model→llm_dim) 替换 ReprogrammingLayer，去掉 prototype cross-attn
--no-backbone   诊断消融：完全跳过 LLM backbone，reprogram 输出直接池化送 regressor；
                若 MAE 与正常 lora 相近，说明 backbone 没有真正参与预测
--random-init   诊断消融：backbone 用随机权重（结构同 Qwen 但无预训练知识）；
                若 MAE 与正常 lora 相近，说明预训练语义没被利用
--dynamic-prompt 动态文本 prompt：把当前窗口统计量（深度/T-S趋势/T-S相关性）用自然语言写进
                prompt，让输入分布更靠文本以激活预训练语义；reprogram numeric 路径不变
"""

import argparse as _ap
import os

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
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import configs.reprogram_tsf as cfg
from models.llm_backbone import load_backbone
from utils import apply_gpu_arg, apply_samples_arg
from models.reprogram_backbone import ThermoForecasterReprogram
from trainers.tsf_trainer import TSFTrainer, build_datasets

# ── 深度归一化（target 用）─────────────────────────────────────────────────────
_D_RANGE = cfg.TSF_DEPTH_MAX - cfg.TSF_DEPTH_MIN

def _norm_target(z):
    return (z - cfg.TSF_DEPTH_MIN) / _D_RANGE * 2.0 - 1.0


# ── collate_fn ────────────────────────────────────────────────────────────────

def collate_fn(batch, is_train=False):
    """
    将 CTDTSFDataset 样本转为原始张量，不经过 tokenizer。

    返回：
      depths:     (B, K, 1)   AUV 深度（米，负值）
      T_profs:    (B, K, N)   温度局部剖面
      S_profs:    (B, K, N)   盐度局部剖面
      season_ids: (B,)        季节索引 0-3（用于生成文字 prompt）
      targets:    (B, H)      归一化目标深度 [-1, 1]
    """
    depths_list, T_list, S_list, ctx_list, sid_list, tgt_list = [], [], [], [], [], []

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

        depths_t = torch.tensor([[float(d)] for d in input_depths],
                                dtype=torch.float32)             # (K, 1)
        T_t = torch.tensor(input_feats, dtype=torch.float32)    # (K, N)
        S_t = (torch.nan_to_num(torch.tensor(input_sal, dtype=torch.float32), nan=0.0)
               if has_sal
               else torch.zeros(K, N, dtype=torch.float32))     # (K, N)

        season_norm = float(ex["season_id"]) / 3.0

        # 深度统计（归一化到 [-1,1]，与 reprogram_backbone._normalize_inputs 一致）
        d_range = cfg.TSF_DEPTH_MAX - cfg.TSF_DEPTH_MIN
        depths_arr = [float(d) for d in input_depths]
        depth_mean_n = (float(np.mean(depths_arr)) - cfg.TSF_DEPTH_MIN) / d_range * 2.0 - 1.0
        depth_range_n = (float(max(depths_arr)) - float(min(depths_arr))) / (-cfg.TSF_DEPTH_MIN) * 2.0 - 1.0

        # T/S 统计量（全局 z-score，clip 到 [-3,3]）
        T_arr = np.array(input_feats, dtype=np.float32)
        T_mean_n  = float(np.clip(T_arr.mean() / (T_arr.std() + 1e-6), -3, 3)) / 3.0
        T_std_n   = float(np.clip(T_arr.std() / (T_arr.mean() + 1e-6 + 10.0), -3, 3)) / 3.0
        if has_sal:
            S_arr = np.array(input_sal, dtype=np.float32)
            S_mean_n = float(np.clip(S_arr.mean() / (S_arr.std() + 1e-6), -3, 3)) / 3.0
            S_std_n  = float(np.clip(S_arr.std() / (S_arr.mean() + 1e-6 + 30.0), -3, 3)) / 3.0
        else:
            S_mean_n = S_std_n = 0.0

        ctx_t = torch.tensor(
            [season_norm, float(ex["doy_sin"]), float(ex["doy_cos"]),
             depth_mean_n, depth_range_n, T_mean_n, T_std_n, S_mean_n, S_std_n],
            dtype=torch.float32,
        )                                                        # (9,)

        tgt_t = torch.tensor(
            [_norm_target(float(z)) for z in ex["target_depths"]],
            dtype=torch.float32,
        )                                                        # (H,)

        depths_list.append(depths_t)
        T_list.append(T_t)
        S_list.append(S_t)
        ctx_list.append(ctx_t)
        sid_list.append(int(ex["season_id"]))
        tgt_list.append(tgt_t)

    return (
        torch.stack(depths_list),                                      # (B, K, 1)
        torch.stack(T_list),                                           # (B, K, N)
        torch.stack(S_list),                                           # (B, K, N)
        torch.stack(ctx_list),                                         # (B, 3)
        torch.tensor(sid_list, dtype=torch.long),                      # (B,)
        torch.stack(tgt_list),                                         # (B, H)
    )


def collate_train(batch): return collate_fn(batch, is_train=True)
def collate_val(batch):   return collate_fn(batch, is_train=False)


# ── forward_fn ────────────────────────────────────────────────────────────────

def forward_fn(model, batch):
    depths, T_profs, S_profs, context, season_ids, targets = batch
    preds = model(depths, T_profs, S_profs, context, season_ids)
    loss  = F.huber_loss(preds, targets, delta=cfg.HUBER_DELTA)
    if not torch.isfinite(loss):
        print(f"[WARN] non-finite loss={loss.item():.4f}, skipping batch")
        loss = torch.zeros(1, device=loss.device, requires_grad=True).squeeze()
    return loss, preds, targets


# ── save_fn ────────────────────────────────────────────────────────────────────

def make_save_fn(mode: str):
    def save_fn(model, run_id, epoch, val_mae, ckpt_dir, is_best):
        non_backbone = {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("backbone.")
        }
        if is_best:
            if mode == "lora":
                model.backbone.save_pretrained(cfg.TSF_LORA_BEST)
            torch.save({"reprogram": non_backbone}, cfg.TSF_HEAD_BEST)
            print(f"  → 最优模型已保存至 {cfg.TSF_HEAD_BEST}")
        else:
            head_path = os.path.join(ckpt_dir, "head_latest.pt")
            if mode == "lora":
                lora_dir = os.path.join(ckpt_dir, "lora_latest")
                if os.path.exists(lora_dir):
                    shutil.rmtree(lora_dir)
                os.makedirs(lora_dir, exist_ok=True)
                model.backbone.save_pretrained(lora_dir)
            torch.save({"reprogram": non_backbone}, head_path)
            print(f"  → 检查点已保存: {ckpt_dir}  (ep{epoch} mae={val_mae:.2f}m)")
    return save_fn


# ── 主流程 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu",  type=str, default=None)
    parser.add_argument("--mode", type=str, default="lora", choices=["lora", "head"],
                        help="lora: LoRA+reprogram+head；head: 冻结 LLM 仅训 reprogram+head（LR/WD/schedule 与 lora 相同）")
    parser.add_argument("--no-reprogram", action="store_true",
                        help="消融：用 Linear 替换 ReprogrammingLayer，去掉 prototype cross-attn")
    parser.add_argument("--no-backbone", action="store_true",
                        help="诊断消融：完全跳过 LLM backbone，reprogram 输出直接池化送 regressor，"
                             "验证 backbone（预训练/随机）是否真的参与预测")
    parser.add_argument("--random-init", action="store_true",
                        help="诊断消融：backbone 用随机权重（结构同 Qwen 但无预训练知识），"
                             "对比真预训练验证 LLM 先验是否起作用")
    parser.add_argument("--dynamic-prompt", action="store_true",
                        help="动态文本 prompt：把当前窗口统计量（深度/T-S趋势/T-S相关性）"
                             "用自然语言写进 prompt，让输入分布更靠文本以激活预训练语义；"
                             "reprogram numeric 路径不变。建议与 --random-init 对照验证")
    args, _ = parser.parse_known_args()

    apply_gpu_arg(args.gpu or cfg.CUDA_DEVICE_LLM)
    apply_samples_arg(cfg)  # 解析 --samples，覆盖 cfg.MAX_SAMPLES_PER_EPOCH
    accelerator = Accelerator(gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS)

    random.seed(cfg.MANUAL_SEED)
    np.random.seed(cfg.MANUAL_SEED)
    torch.manual_seed(cfg.MANUAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.MANUAL_SEED)

    train_dataset, val_dataset, baselines = build_datasets(cfg)

    use_reprogram = not args.no_reprogram
    use_backbone  = not args.no_backbone
    # no-backbone 时 backbone 不参与前向，LoRA 无意义 → 强制按 head 语义（冻结 backbone）
    effective_mode = "head" if args.no_backbone else args.mode
    if args.no_backbone and args.mode == "lora":
        print("[no-backbone] backbone 被跳过，LoRA 无效 → 自动切换为 head 模式（不训练/不保存 backbone）")

    peft_model, tokenizer = load_backbone(cfg, random_init=args.random_init)
    forecaster = ThermoForecasterReprogram(
        peft_model     = peft_model,
        tokenizer      = tokenizer,
        H              = cfg.TSF_H,
        K              = cfg.TSF_K,
        N              = cfg.SAMPLE_WINDOW_SIZE,
        d_model        = cfg.REPROGRAM_D_MODEL,
        n_heads        = cfg.REPROGRAM_N_HEADS,
        n_prototypes   = cfg.REPROGRAM_N_PROTOTYPES,
        depth_min      = cfg.TSF_DEPTH_MIN,
        depth_max      = cfg.TSF_DEPTH_MAX,
        dropout        = cfg.REPROGRAM_DROPOUT,
        use_reprogram  = use_reprogram,
        use_backbone   = use_backbone,
        use_dynamic_prompt = args.dynamic_prompt,
    )

    # head 模式（或 no-backbone）：冻结整个 backbone，只训练 reprogram 层和 regressor
    if effective_mode == "head":
        for p in forecaster.backbone.parameters():
            p.requires_grad_(False)
        print(f"[mode={effective_mode}] backbone 已冻结，只训练 reprogram 层 + regressor")

    backbone_n = sum(p.numel() for p in forecaster.backbone.parameters() if p.requires_grad)
    if effective_mode == "lora":
        print(f"[check] backbone trainable params: {backbone_n:,}  (应 >0 才说明 LoRA 生效)")
    else:
        print(f"[check] backbone trainable params: {backbone_n:,}  (冻结/no-backbone 应为 0)")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_train, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_val,   num_workers=0)

    reprogram_params = (
        list(forecaster.depth_proj.parameters())
        + list(forecaster.T_proj.parameters())
        + list(forecaster.S_proj.parameters())
        + list(forecaster.ctx_proj.parameters())
        + list(forecaster.embed_norm.parameters())
    )
    if args.no_reprogram:
        reprogram_params += (
            list(forecaster.linear_depth.parameters())
            + list(forecaster.linear_T.parameters())
            + list(forecaster.linear_S.parameters())
            + list(forecaster.linear_ctx.parameters())
        )
    else:
        reprogram_params += (
            list(forecaster.reprogram_depth.parameters())
            + list(forecaster.reprogram_T.parameters())
            + list(forecaster.reprogram_S.parameters())
            + list(forecaster.reprogram_ctx.parameters())
            + [forecaster.prototypes]
        )
    head_params = list(forecaster.regressor.parameters())

    # reprogram + head 参数组：lora / head 两种模式共用同一套 LR、WD（公平消融）
    reprogram_head_groups = [
        {"params": reprogram_params, "lr": cfg.LR_REPROGRAM, "weight_decay": cfg.WEIGHT_DECAY, "name": "reprogram"},
        {"params": head_params,      "lr": cfg.LR_HEAD,      "weight_decay": cfg.WEIGHT_DECAY, "name": "head"},
    ]
    if effective_mode == "lora":
        lora_params = [p for p in forecaster.backbone.parameters() if p.requires_grad]
        param_groups = [
            {"params": lora_params, "lr": cfg.LR_LORA, "weight_decay": 0.0, "name": "lora"},
            *reprogram_head_groups,
        ]
    else:
        param_groups = reprogram_head_groups
    optimizer = AdamW(param_groups)

    rh_n = sum(p.numel() for g in reprogram_head_groups for p in g["params"])
    print(f"[check] reprogram+head trainable params: {rh_n:,}  "
          f"(lr_reprogram={cfg.LR_REPROGRAM}, lr_head={cfg.LR_HEAD}, wd={cfg.WEIGHT_DECAY})")

    total_steps  = len(train_loader) * cfg.EPOCHS // cfg.GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * getattr(cfg, "WARMUP_RATIO", 0.05))
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        num_cycles=2,
    )

    # ── variant_tag：集中所有区分性标识（mode / no-reprogram / random-init / samples）──
    # 区分信息放在最前面，避免 SwanLab 图例对长公共前缀做中间省略（"qwen3.5-...f-16k-lora"）
    # 导致 -randinit、-no-reprogram 等关键尾缀被截掉看不见。
    samples_k = cfg.MAX_SAMPLES_PER_EPOCH // 1000
    variant_parts = [effective_mode]                             # lora / head（no-backbone 已归一为 head）
    if not use_backbone:    variant_parts.append("nobackbone")   # 跳过 LLM backbone 诊断
    if not use_reprogram:   variant_parts.append("noreprog")     # w/o reprogram 消融
    if args.random_init:    variant_parts.append("randinit")     # 随机初始化诊断
    if args.dynamic_prompt: variant_parts.append("dynprompt")    # 动态文本 prompt
    variant_tag = "-".join(variant_parts)                        # e.g. "head-nobackbone"

    # run name：区分 tag 前置 + 短后缀，公共前缀最短
    cfg.SWANLAB_EXPERIMENT = f"{variant_tag}-{samples_k}k-reprog"

    # 检查点按 variant 分子目录，不同模式互不覆盖
    cfg.TSF_CKPT_DIR  = os.path.join(cfg.TSF_CKPT_ROOT, f"{variant_tag}-{samples_k}k")
    cfg.TSF_LORA_BEST = os.path.join(cfg.TSF_CKPT_DIR, "lora_best")
    cfg.TSF_HEAD_BEST = os.path.join(cfg.TSF_CKPT_DIR, "head_best.pt")
    os.makedirs(cfg.TSF_CKPT_DIR, exist_ok=True)
    print(f"[variant] {variant_tag}  →  run='{cfg.SWANLAB_EXPERIMENT}'  ckpt='{cfg.TSF_CKPT_DIR}'")

    trainer = TSFTrainer(
        cfg          = cfg,
        model        = forecaster,
        optimizer    = optimizer,
        scheduler    = scheduler,
        train_loader = train_loader,
        val_loader   = val_loader,
        accelerator  = accelerator,
        forward_fn   = forward_fn,
        save_fn      = make_save_fn(effective_mode),
        baseline_metrics  = baselines,
        swanlab_extra_cfg = {
            "model":           "Qwen3.5-2B",
            "input_mode":      "reprogram",
            "variant":         variant_tag,
            "train_mode":      effective_mode,
            "use_reprogram":   use_reprogram,
            "use_backbone":    use_backbone,
            "random_init":     args.random_init,
            "dynamic_prompt":  args.dynamic_prompt,
            "K":               cfg.TSF_K,
            "H":               cfg.TSF_H,
            "n_tokens":        f"~50+{3 * cfg.TSF_K} (prompt+[THERMO]→ctx+numeric)",
            "d_model":         cfg.REPROGRAM_D_MODEL,
            "n_heads":         cfg.REPROGRAM_N_HEADS,
            "n_prototypes":    cfg.REPROGRAM_N_PROTOTYPES,
            "lora_r":          cfg.LORA_R,
            "lr_lora":         cfg.LR_LORA if args.mode == "lora" else 0.0,
            "lr_reprogram":    cfg.LR_REPROGRAM,
            "lr_head":         cfg.LR_HEAD,
            "batch_size":      cfg.BATCH_SIZE,
            "grad_accum":      cfg.GRAD_ACCUM_STEPS,
            "epochs":          cfg.EPOCHS,
            "warmup_steps":    warmup_steps,
            "total_steps":     total_steps,
            "weight_decay":    cfg.WEIGHT_DECAY,
            "huber_delta":     cfg.HUBER_DELTA,
            "es_patience":     cfg.ES_PATIENCE,
            "seed":            cfg.MANUAL_SEED,
            "warmup_ratio":    getattr(cfg, "WARMUP_RATIO", 0.05),
            "grad_clip_max":   getattr(cfg, "GRAD_CLIP_MAX_NORM", 1.0),
        },
    )
    trainer.train()
