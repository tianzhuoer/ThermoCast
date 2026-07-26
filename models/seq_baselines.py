"""
LSTM、Transformer、iTransformer 时序预测基线模型。

输入:  (B, K, N_INPUT_FEATURES=8) 统一特征张量（LSTM/Transformer）
       或四路原始剖面（iTransformer，与 Reprogram 对齐）
输出:  (B, H) 归一化温跃层深度预测

回归头统一使用三层 MLP（与 ThermoForecasterReprogram.regressor 结构对齐）。
"""

import math
import torch
import torch.nn as nn


def _build_head(in_dim: int, H: int) -> nn.Sequential:
    """三层 MLP 回归头，与 ThermoForecasterReprogram.regressor 结构对齐。"""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 64),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(64, H),
    )


class LSTMForecaster(nn.Module):
    """
    双层 LSTM + 三层 MLP 回归头。

    取最后一步的 hidden state 作为序列表示，经 MLP 输出预测。
    """

    def __init__(self, input_size: int = 8, hidden_size: int = 128,
                 num_layers: int = 2, H: int = 5, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _build_head(hidden_size, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, input_size) → (B, H)"""
        out, _ = self.lstm(x)          # (B, K, hidden_size)
        return self.head(out[:, -1])   # last step → (B, H)


class TransformerForecaster(nn.Module):
    """
    轻量 Transformer Encoder-only baseline（Pre-LN）+ 线性输出头。

    作为弱 baseline 与 Reprogram 对比：d_model=64，2层，8维压缩输入，无噪声增强。
    """

    def __init__(self, input_size: int = 8, d_model: int = 64, n_heads: int = 2,
                 num_layers: int = 2, ffn_dim: int = 128, H: int = 5,
                 dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, input_size) → (B, H)"""
        x = self.input_proj(x) + self.pe[:, :x.size(1)]
        x = self.encoder(x)
        return self.head(x[:, -1])


class iTransformerForecaster(nn.Module):
    """
    iTransformer（ICLR 2024）：倒置 Transformer 时序预测基线。

    输入与 ThermoForecasterReprogram 对齐：四路原始剖面，内部做归一化。
      depths:  (B, K, 1)   AUV 深度
      T_profs: (B, K, N)   温度剖面
      S_profs: (B, K, N)   盐度剖面
      context: (B, 3)      [season_norm, doy_sin, doy_cos]

    前向流程：
      拼接 → (B, K, 1+N+N+3) → 转置 → (B, 1+N+N+3, K)
      → Embedding: Linear(K → d_model)
      → L 层 Encoder（Attention 跨 variate token）
      → mean pool: (B, d_model)
      → 三层 MLP head → (B, H)
    """

    def __init__(self, K: int, N: int = 15, d_model: int = 128,
                 n_heads: int = 4, num_layers: int = 2, ffn_dim: int = 256,
                 H: int = 5, dropout: float = 0.1,
                 depth_min: float = -150.0, depth_max: float = -25.0):
        super().__init__()
        self.register_buffer("depth_min", torch.tensor(depth_min))
        self.register_buffer("depth_max", torch.tensor(depth_max))

        # 1(depth) + N(T) + N(S) + 3(ctx) 个 variate，每个 variate 有 K 步序列
        n_variates = 1 + N + N + 3
        self.embedding = nn.Linear(K, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = _build_head(d_model, H)

    def _normalize(self, depths, T_profs, S_profs):
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
        → (B, H)
        """
        B, K, _ = depths.shape
        depths_n, T_n, S_n = self._normalize(depths, T_profs, S_profs)
        ctx_exp = context.unsqueeze(1).expand(B, K, 3)          # (B, K, 3)
        x = torch.cat([depths_n, T_n, S_n, ctx_exp], dim=2)    # (B, K, 1+N+N+3)
        x = x.permute(0, 2, 1)                                  # (B, 1+N+N+3, K)
        x = self.embedding(x)                                    # (B, n_variates, d_model)
        x = self.encoder(x)                                      # (B, n_variates, d_model)
        x = x.mean(dim=1)                                        # (B, d_model)
        return self.head(x)                                      # (B, H)


def load_lstm(cfg):
    model = LSTMForecaster(
        input_size=cfg.RAW_INPUT_SIZE,
        hidden_size=cfg.LSTM_HIDDEN_SIZE,
        num_layers=cfg.LSTM_NUM_LAYERS,
        H=cfg.TSF_H,
        dropout=cfg.LSTM_DROPOUT,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"LSTMForecaster loaded. Params: {n:,}")
    return model


def load_transformer(cfg):
    model = TransformerForecaster(
        input_size=cfg.RAW_INPUT_SIZE,
        d_model=cfg.TF_D_MODEL,
        n_heads=cfg.TF_N_HEADS,
        num_layers=cfg.TF_NUM_LAYERS,
        ffn_dim=cfg.TF_FFN_DIM,
        H=cfg.TSF_H,
        dropout=cfg.TF_DROPOUT,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"TransformerForecaster loaded. Params: {n:,}")
    return model


def load_itransformer(cfg):
    model = iTransformerForecaster(
        K=cfg.TSF_K,
        N=cfg.SAMPLE_WINDOW_SIZE,
        d_model=cfg.IT_D_MODEL,
        n_heads=cfg.IT_N_HEADS,
        num_layers=cfg.IT_NUM_LAYERS,
        ffn_dim=cfg.IT_FFN_DIM,
        H=cfg.TSF_H,
        dropout=cfg.IT_DROPOUT,
        depth_min=cfg.TSF_DEPTH_MIN,
        depth_max=cfg.TSF_DEPTH_MAX,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"iTransformerForecaster loaded. Params: {n:,}")
    return model
