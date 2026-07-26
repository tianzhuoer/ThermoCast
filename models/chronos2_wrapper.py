import torch
import torch.nn as nn
from chronos import Chronos2Pipeline

# Chronos-2 每个 output patch 固定输出 16 个时间步
_CHRONOS2_PATCH_STEPS = 16

# 送入 backbone 的时序模态数：depth, T_prof, S_prof
# context（season/doy）为时不变量，直接绕过 backbone 拼到 head
_N_MODALITIES = 3


def load_chronos2(cfg):
    wrapper = Chronos2Wrapper(
        model_id=cfg.CHRONOS2_MODEL_ID,
        cache_dir=cfg.CHRONOS2_CACHE_DIR,
        H=cfg.TSF_H,
        N=getattr(cfg, 'TSF_N_PROFILE', 15),
        freeze_backbone=getattr(cfg, 'FREEZE_BACKBONE', False),
        depth_min=cfg.TSF_DEPTH_MIN,
        depth_max=cfg.TSF_DEPTH_MAX,
    )
    print(f"Chronos-2 loaded. Trainable params: "
          f"{sum(p.numel() for p in wrapper.parameters() if p.requires_grad):,}")
    return wrapper


class Chronos2Wrapper(nn.Module):
    """
    Chronos-2 多模态微调包装器，输入风格与 ThermoForecasterReprogram 对齐。

    输入:
      depths:  (B, K, 1)   AUV 深度（米，负值）
      T_profs: (B, K, N)   温度局部剖面
      S_profs: (B, K, N)   盐度局部剖面
      context: (B, 3)      [season_norm, doy_sin, doy_cos]

    三条时序模态（depth/T/S）各经 per-modality Linear 压缩为标量序列 (B, K)，
    用 group_ids 送入 backbone 激活 GroupSelfAttention 实现跨模态交互。
    context 为时不变量（std≈0），绕过 backbone 直接拼接至 head，
    避免 Chronos-2 内部 instance normalization 除以 0 产生 nan。

    骨干输出 quantile_preds: (B*3, 21, 16)
    median (分位数 10) → (B*3, 16) → reshape (B, 48) → cat context (B, 3)
    → (B, 51) → projection_head → (B, H)
    """

    def __init__(self, model_id: str, cache_dir: str, H: int, N: int = 15,
                 freeze_backbone: bool = False,
                 depth_min: float = -150.0, depth_max: float = -25.0):
        super().__init__()
        pipeline = Chronos2Pipeline.from_pretrained(
            model_id, cache_dir=cache_dir, dtype=torch.float32,
            local_files_only=True,
        )
        self.base = pipeline.model
        self.H    = H

        # per-modality projections → 标量序列（供 Chronos-2 消费）
        # depth 已归一化为标量，无需 Linear；T/S 需从 N 维压缩
        self.T_proj = nn.Linear(N, 1)
        self.S_proj = nn.Linear(N, 1)

        # 3 个时序模态 × 16 patch steps + 3 维 context
        head_in = _N_MODALITIES * _CHRONOS2_PATCH_STEPS + 3  # 3*16+3 = 51
        self.projection_head = nn.Linear(head_in, H)

        self.register_buffer("depth_min", torch.tensor(depth_min))
        self.register_buffer("depth_max", torch.tensor(depth_max))

        if freeze_backbone:
            for p in self.base.parameters():
                p.requires_grad = False

    def _normalize_inputs(self, depths, T_profs, S_profs):
        d_range  = self.depth_max - self.depth_min
        depths_n = (depths - self.depth_min) / d_range * 2.0 - 1.0
        T_n = (T_profs - T_profs.mean(2, keepdim=True)) / (T_profs.std(2, keepdim=True) + 1e-6)
        S_n = (S_profs - S_profs.mean(2, keepdim=True)) / (S_profs.std(2, keepdim=True) + 1e-6)
        return depths_n, T_n, S_n

    def forward(self, depths: torch.Tensor, T_profs: torch.Tensor,
                S_profs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        depths:  (B, K, 1)
        T_profs: (B, K, N)
        S_profs: (B, K, N)
        context: (B, 3)
        返回: (B, H)
        """
        B, K, _ = depths.shape

        depths_n, T_n, S_n = self._normalize_inputs(depths, T_profs, S_profs)

        # 时序模态 → (B, K) 标量序列
        d_seq = depths_n.squeeze(-1)          # (B, K)，已在 [-1,1]，无需额外 proj
        T_seq = self.T_proj(T_n).squeeze(-1)  # (B, K)
        S_seq = self.S_proj(S_n).squeeze(-1)  # (B, K)

        # 拼成 (B*3, K)，group_ids 标注同一样本
        flat      = torch.stack([d_seq, T_seq, S_seq], dim=1).reshape(B * _N_MODALITIES, K)
        group_ids = torch.arange(B, device=flat.device).repeat_interleave(_N_MODALITIES)

        out    = self.base(context=flat, group_ids=group_ids, num_output_patches=1)
        median = out.quantile_preds[:, 10, :]                        # (B*3, 16)
        median = median.reshape(B, _N_MODALITIES * _CHRONOS2_PATCH_STEPS)  # (B, 48)

        # context 时不变，直接拼接绕过 instance norm
        fused = torch.cat([median, context], dim=-1)                 # (B, 51)
        return self.projection_head(fused)                           # (B, H)
