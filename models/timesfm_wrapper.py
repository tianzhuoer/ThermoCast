import torch
import torch.nn as nn
from transformers import TimesFm2_5ModelForPrediction

# 送入 backbone 的时序模态数：depth, T_prof, S_prof
# context（season/doy）为时不变量，直接绕过 backbone 拼到 head
_N_MODALITIES = 3


def load_timesfm(cfg):
    wrapper = TimesFMWrapper(
        model_id=cfg.TIMESFM_MODEL_ID,
        cache_dir=cfg.TIMESFM_CACHE_DIR,
        H=cfg.TSF_H,
        input_len=cfg.TIMESFM_INPUT_LEN,
        N=getattr(cfg, 'TSF_N_PROFILE', 15),
        freeze_backbone=getattr(cfg, 'FREEZE_BACKBONE', False),
        depth_min=cfg.TSF_DEPTH_MIN,
        depth_max=cfg.TSF_DEPTH_MAX,
    )
    print(f"TimesFM loaded. Trainable params: "
          f"{sum(p.numel() for p in wrapper.parameters() if p.requires_grad):,}")
    return wrapper


class TimesFMWrapper(nn.Module):
    """
    TimesFM 2.5 多模态微调包装器，输入风格与 Chronos2Wrapper/ThermoForecasterReprogram 对齐。

    输入:
      depths:  (B, K, 1)   AUV 深度（米，负值）
      T_profs: (B, K, N)   温度局部剖面
      S_profs: (B, K, N)   盐度局部剖面
      context: (B, 3)      [season_norm, doy_sin, doy_cos]

    三条时序模态（depth/T/S）各经 per-modality Linear 压缩为标量序列 (B, K)，
    拼成 (B*3, K) 后批量送入 TimesFM backbone（TimesFM 不支持 group_ids，
    以独立单变量序列处理）。context 时不变，绕过 backbone 直接拼到 head。

    backbone 输出 mean_predictions: (B*3, T_out)
    → reshape (B, 3*T_out) → cat context (B, 3) → projection_head → (B, H)
    """

    def __init__(self, model_id: str, cache_dir: str, H: int, input_len: int,
                 N: int = 15, freeze_backbone: bool = False,
                 depth_min: float = -150.0, depth_max: float = -25.0):
        super().__init__()
        self.H         = H
        self.input_len = input_len

        # per-modality projections → 标量序列
        # depth 已归一化为标量，T/S 从 N 维压缩
        self.T_proj = nn.Linear(N, 1)
        self.S_proj = nn.Linear(N, 1)

        self.base = TimesFm2_5ModelForPrediction.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        ).to(torch.float32)

        if freeze_backbone:
            for p in self.base.parameters():
                p.requires_grad_(False)

        # dummy forward 确定 backbone 单条序列的输出步数
        with torch.no_grad():
            dummy_out = self.base(
                past_values=[torch.zeros(input_len)],
                forecast_context_len=input_len,
            )
            self._base_out_steps = dummy_out.mean_predictions.shape[1]

        # 3 个模态各输出 T_out 步 + 3 维 context
        head_in = _N_MODALITIES * self._base_out_steps + 3
        self.projection_head = nn.Linear(head_in, H)

        self.register_buffer("depth_min", torch.tensor(depth_min))
        self.register_buffer("depth_max", torch.tensor(depth_max))

    def _normalize_inputs(self, depths, T_profs, S_profs):
        d_range  = self.depth_max - self.depth_min
        depths_n = (depths - self.depth_min) / d_range * 2.0 - 1.0
        T_n = (T_profs - T_profs.mean(2, keepdim=True)) / (T_profs.std(2, keepdim=True) + 1e-6)
        S_n = (S_profs - S_profs.mean(2, keepdim=True)) / (S_profs.std(2, keepdim=True) + 1e-6)
        return depths_n, T_n, S_n

    def _pad_or_trim(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (M, K) → (M, input_len)，左侧补零或右侧截断。"""
        M, K = seq.shape
        L = self.input_len
        if K < L:
            pad = torch.zeros(M, L - K, device=seq.device, dtype=seq.dtype)
            return torch.cat([pad, seq], dim=1)
        return seq[:, -L:]

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
        d_seq = depths_n.squeeze(-1)          # (B, K)
        T_seq = self.T_proj(T_n).squeeze(-1)  # (B, K)
        S_seq = self.S_proj(S_n).squeeze(-1)  # (B, K)

        # 拼成 (B*3, K)，pad/trim 到 input_len
        flat = torch.stack([d_seq, T_seq, S_seq], dim=1).reshape(B * _N_MODALITIES, K)
        flat = self._pad_or_trim(flat)  # (B*3, input_len)

        # TimesFM 接受 list of 1D tensors
        pv_list = [flat[i] for i in range(B * _N_MODALITIES)]
        outputs = self.base(past_values=pv_list, forecast_context_len=self.input_len)

        preds = outputs.mean_predictions                          # (B*3, T_out)
        preds = preds.reshape(B, _N_MODALITIES * self._base_out_steps)  # (B, 3*T_out)

        # context 时不变，直接拼接
        fused = torch.cat([preds, context], dim=-1)              # (B, 3*T_out+3)
        return self.projection_head(fused)                       # (B, H)
