"""
TSFTrainer：温跃层时序预测（TSF）通用训练器。

任何符合以下接口的 TSF 模型均可使用本 Trainer：
  forward_fn(model, batch) -> (loss, preds_norm, targets_norm)
    - preds_norm / targets_norm 归一化到 [-1, 1]（由 cfg.TSF_DEPTH_MIN/MAX 定义）
    - 训练指标（MAE/RMSE）由 Trainer 在反归一化后以米为单位计算

  save_fn(unwrapped_model, run_id, epoch, val_mae, ckpt_dir, is_best)
    - 模型特定的保存逻辑；is_best=True 时保存至 cfg.TSF_BEST_DIR / TSF_LORA_BEST

目前已支持：
  - LLM-TSF（ThermoForecaster）   → train_llm_tsf.py
  - TimesFM fine-tune             → train_timesfm.py
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable

import numpy as np
import swanlab
import torch
from torch.utils.data import Subset

from tsf_data.metrics import compute_metrics


# ── 数据集构建 ─────────────────────────────────────────────────────────────────

def build_datasets(cfg):
    """
    构建训练/验证数据集，应用 MAX_STEPS_PER_EPOCH 截断，返回基线指标。

    返回 (train_dataset, val_dataset, baselines)。
    所有训练脚本共用此函数，确保数据划分与截断逻辑完全一致。
    """
    import os
    from tsf_data.tsf_dataset import CTDTSFDataset

    # 多进程时只让 rank 0 打印，其余进程静默加载
    verbose = (os.environ.get('LOCAL_RANK', '0') == '0')

    cache_dir = getattr(cfg, 'DATASET_CACHE_DIR', None)
    train_dataset = CTDTSFDataset(split='train',      seed=cfg.MANUAL_SEED, verbose=verbose, cache_dir=cache_dir)
    val_dataset   = CTDTSFDataset(split='validation', seed=cfg.MANUAL_SEED, verbose=verbose, cache_dir=cache_dir)
    baselines     = compute_baseline_metrics(train_dataset, val_dataset)

    max_samples = getattr(cfg, 'MAX_SAMPLES_PER_EPOCH', None)
    if max_samples and max_samples < len(train_dataset):
        idx = torch.randperm(
            len(train_dataset),
            generator=torch.Generator().manual_seed(cfg.MANUAL_SEED),
        )[:max_samples]
        train_dataset = Subset(train_dataset, idx.tolist())

    return train_dataset, val_dataset, baselines


# ── 基线指标（训练开始时记录，作为参照）────────────────────────────────────────

def compute_baseline_metrics(train_dataset, val_dataset) -> dict:
    """用训练集目标均值构造全局/季节基线，在验证集上计算（单位：米）。"""
    train_targets  = np.asarray([s['target_depths'] for s in train_dataset], dtype=np.float32)
    val_targets    = np.asarray([s['target_depths'] for s in val_dataset],   dtype=np.float32)
    train_seasons  = np.asarray([int(s['season_id']) for s in train_dataset], dtype=np.int64)
    val_seasons    = np.asarray([int(s['season_id']) for s in val_dataset],   dtype=np.int64)

    global_pred   = train_targets.mean(axis=0)
    season_means  = {
        sid: train_targets[train_seasons == sid].mean(axis=0)
        for sid in sorted(set(train_seasons.tolist()))
    }
    season_pred = np.stack([
        season_means.get(int(sid), global_pred) for sid in val_seasons
    ], axis=0)

    g_err = global_pred[None, :] - val_targets
    s_err = season_pred - val_targets
    return {
        "val/baseline_global_mae":  float(np.abs(g_err).mean()),
        "val/baseline_global_rmse": float(np.sqrt(np.square(g_err).mean())),
        "val/baseline_season_mae":  float(np.abs(s_err).mean()),
        "val/baseline_season_rmse": float(np.sqrt(np.square(s_err).mean())),
    }


# ── 训练器 ──────────────────────────────────────────────────────────────────────

class TSFTrainer:
    """
    通用 TSF 训练器。

    参数
    ----
    cfg          : 配置模块（需含 EPOCHS, GRAD_ACCUM_STEPS, CKPT_INTERVAL,
                   ES_PATIENCE, TSF_DEPTH_MIN, TSF_DEPTH_MAX,
                   SWANLAB_PROJECT, SWANLAB_EXPERIMENT, SWANLAB_LOG_INTERVAL,
                   TSF_CKPT_DIR, TSF_BEST_DIR or TSF_LORA_BEST）
    model        : nn.Module（未经 accelerator.prepare 的原始模型）
    optimizer    : 未经 prepare 的优化器
    scheduler    : 未经 prepare 的 LR scheduler
    train_loader : 未经 prepare 的 DataLoader
    val_loader   : 未经 prepare 的 DataLoader
    accelerator  : Accelerator 实例
    forward_fn   : (model, batch) -> (loss, preds_norm, targets_norm)
    save_fn      : (unwrapped_model, run_id, epoch, val_mae, ckpt_dir, is_best) -> None
    baseline_metrics : 由 compute_baseline_metrics 预计算的基线字典（可选）
    swanlab_extra_cfg: 额外写入 SwanLab config 的字典（可选）
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        accelerator,
        forward_fn:    Callable,
        save_fn:       Callable,
        baseline_metrics: dict | None = None,
        swanlab_extra_cfg: dict | None = None,
    ):
        self.cfg         = cfg
        self.model       = model
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.acc         = accelerator
        self.forward_fn  = forward_fn
        self.save_fn     = save_fn
        self.baselines   = baseline_metrics or {}
        self.extra_cfg   = swanlab_extra_cfg or {}

        d_range          = cfg.TSF_DEPTH_MAX - cfg.TSF_DEPTH_MIN
        self._d_range    = d_range
        self._d_min      = cfg.TSF_DEPTH_MIN

    # ── 深度归一化 ──────────────────────────────────────────────────────────────

    def _denorm(self, z_norm):
        """归一化深度 [-1,1] → 米"""
        return (z_norm + 1.0) / 2.0 * self._d_range + self._d_min

    # ── 主训练流程 ──────────────────────────────────────────────────────────────

    def train(self):
        cfg  = self.cfg
        acc  = self.acc
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Accelerate 准备 ──────────────────────────────────────────────────
        (self.model, self.optimizer,
         self.train_loader, self.val_loader,
         self.scheduler) = acc.prepare(
            self.model, self.optimizer,
            self.train_loader, self.val_loader,
            self.scheduler,
        )

        # ── SwanLab 初始化 ──────────────────────────────────────────────────
        if acc.is_main_process:
            swanlab.init(
                project=cfg.SWANLAB_PROJECT,
                experiment_name=cfg.SWANLAB_EXPERIMENT,
                config={**self.extra_cfg, **self.baselines},
            )
            if self.baselines:
                print("Baselines on validation:")
                for k, v in self.baselines.items():
                    print(f"  {k}: {v:.2f}m")

        best_val_mae  = float('inf')
        es_no_improve = 0
        should_stop   = False
        global_step   = 0
        ckpt_dir      = getattr(cfg, 'TSF_CKPT_DIR', 'checkpoints_tsf')
        grad_clip_max_norm = float(getattr(cfg, "GRAD_CLIP_MAX_NORM", 1.0))

        for epoch in range(cfg.EPOCHS):
            # 多 GPU：广播 early-stop 信号防止 NCCL 超时
            stop_tensor = torch.tensor(int(should_stop), device=acc.device)
            stop_tensor = acc.gather(stop_tensor)
            if stop_tensor.max().item() > 0:
                break

            # ── 训练 epoch ──────────────────────────────────────────────────
            self.model.train()
            total_huber    = 0.0
            total_sq_err_m = 0.0
            total_samples  = 0
            win_loss = win_samples = 0.0
            last_grad_norm_preclip  = 0.0
            last_grad_norm_postclip = 0.0
            last_group_norms: dict[str, float] = {}

            for i, batch in enumerate(self.train_loader):
                with acc.accumulate(self.model):
                    loss, preds, targets = self.forward_fn(self.model, batch)
                    acc.backward(loss)
                    if acc.sync_gradients:
                        # 裁剪前分组梯度范数（仅对带 name 标签的参数组生效）
                        for g in self.optimizer.param_groups:
                            gname = g.get('name')
                            if gname is None:
                                continue
                            gn = sum(
                                p.grad.data.norm(2).item() ** 2
                                for p in g['params'] if p.grad is not None
                            ) ** 0.5
                            last_group_norms[f"train/grad_norm_{gname}"] = gn

                        last_grad_norm_preclip = float(
                            acc.clip_grad_norm_(
                                acc.unwrap_model(self.model).parameters(),
                                max_norm=grad_clip_max_norm,
                            )
                        )
                        last_grad_norm_postclip = min(last_grad_norm_preclip, grad_clip_max_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                bs = targets.size(0)
                total_huber    += loss.item() * bs
                total_samples  += bs
                with torch.no_grad():
                    sq_m = (self._denorm(preds.detach()) - self._denorm(targets)).pow(2).mean().item()
                total_sq_err_m += sq_m * bs
                win_loss    += loss.item() * bs
                win_samples += bs
                global_step += 1

                if i % cfg.SWANLAB_LOG_INTERVAL == 0:
                    w = win_loss / win_samples
                    if acc.is_main_process:
                        print(f"Epoch [{epoch+1}/{cfg.EPOCHS}] "
                              f"Step [{i+1}/{len(self.train_loader)}] "
                              f"huber={w:.4f}  "
                              f"grad(pre/post)={last_grad_norm_preclip:.3f}/{last_grad_norm_postclip:.3f}")
                        swanlab.log(
                            {
                                "train/huber":              w,
                                "train/grad_norm":          last_grad_norm_preclip,
                                "train/grad_norm_preclip":  last_grad_norm_preclip,
                                "train/grad_norm_postclip": last_grad_norm_postclip,
                                **last_group_norms,
                            },
                            step=global_step,
                        )
                    win_loss = win_samples = 0.0

            train_huber = total_huber / max(total_samples, 1)
            train_rmse  = (total_sq_err_m / max(total_samples, 1)) ** 0.5

            # ── 验证 epoch ──────────────────────────────────────────────────
            self.model.eval()
            all_preds, all_targets = [], []

            with torch.no_grad():
                for batch in self.val_loader:
                    _, preds, targets = self.forward_fn(self.model, batch)
                    p_all, t_all = acc.gather_for_metrics((preds, targets))
                    if acc.is_main_process:
                        all_preds.append(p_all.cpu())
                        all_targets.append(t_all.cpu())

            if not acc.is_main_process:
                continue

            if not all_preds:
                print(f">>> Epoch {epoch+1}: 验证集为空，跳过。")
                continue

            all_preds   = torch.cat(all_preds,   dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            preds_m     = self._denorm(all_preds).cpu().numpy()
            targets_m   = self._denorm(all_targets).cpu().numpy()

            m        = compute_metrics(preds_m, targets_m)
            val_mae  = m["mae"]
            pred_std = float(preds_m.std())

            lr0 = self.scheduler.get_last_lr()[0]
            lr1 = self.scheduler.get_last_lr()[-1]
            step_str = "  ".join(f"h{h+1}={v:.2f}m" for h, v in enumerate(m["step_mae"]))
            print(f">>> Epoch {epoch+1}/{cfg.EPOCHS}  "
                  f"train_huber={train_huber:.4f}  train_rmse={train_rmse:.2f}m  "
                  f"val_mae={val_mae:.2f}m  val_rmse={m['rmse']:.2f}m  "
                  f"smape={m['smape']:.2f}%  mape={m['mape']:.2f}%  mase={m['mase']:.3f}  "
                  f"pred_std={pred_std:.2f}m  [{step_str}]  "
                  f"lr={lr0:.2e}/{lr1:.2e}")

            log_dict = {
                "val/mae":           val_mae,
                "val/rmse":          m["rmse"],
                "val/mse":           m["mse"],
                "val/smape":         m["smape"],
                "val/mape":          m["mape"],
                "val/mase":          m["mase"],
                "train/epoch_huber": train_huber,
                "train/epoch_rmse":  train_rmse,
                "lr/main":           lr0,
            }
            if epoch == 0:
                log_dict.update(self.baselines)
            for h, v in enumerate(m["step_mae"]):
                log_dict[f"val/mae_h{h+1}"] = v
            swanlab.log(log_dict, step=epoch + 1)

            # ── 定期检查点 ──────────────────────────────────────────────────
            if (epoch + 1) % cfg.CKPT_INTERVAL == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                self.save_fn(acc.unwrap_model(self.model),
                             run_id, epoch + 1, val_mae, ckpt_dir, is_best=False)

            # ── 最优模型 + early stopping ──────────────────────────────────
            if val_mae < best_val_mae:
                best_val_mae  = val_mae
                es_no_improve = 0
                os.makedirs(ckpt_dir, exist_ok=True)
                self.save_fn(acc.unwrap_model(self.model),
                             run_id, epoch + 1, val_mae, ckpt_dir, is_best=True)
                print(f">>> 最优模型更新 val_mae={val_mae:.2f}m")
            else:
                es_no_improve += 1
                print(f"    [ES] 无改善 {es_no_improve}/{cfg.ES_PATIENCE}")
                if es_no_improve >= cfg.ES_PATIENCE:
                    es_min = getattr(cfg, 'ES_MIN_EPOCHS', 40)
                    if epoch + 1 < es_min:
                        print(f"    [ES] 达到停止条件，但当前仅第 {epoch+1} epoch，延迟至第 {es_min} epoch 后中断。")
                    else:
                        print(f">>> Early stopping：连续 {cfg.ES_PATIENCE} epoch 无改善。")
                        should_stop = True

        self.acc.wait_for_everyone()
        if self.acc.is_main_process:
            swanlab.finish()
