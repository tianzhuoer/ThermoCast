"""
Time-LLM（ICLR 2024）baseline，适配 LLM_ThermoTSF 统一 pipeline。

定位：**架构对照** baseline —— 与主方法 Reprogram-TSF 同 backbone（Qwen3.5-2B）、
同微调策略（LoRA，复用 models.llm_backbone.load_backbone），仅把架构换成官方
Time-LLM 那套：
  · 时间轴 patch（PatchEmbedding，patch_len/stride）
  · text prototypes：取词表 embedding 切片作可学习 source embedding（见下方说明）
  · ReprogrammingLayer：patch emb cross-attn 对齐到 source embedding
  · Prompt-as-Prefix（PaP）：自动统计量（min/max/median/lags）+ 任务描述文字前缀

与官方差异（仅接入层，核心算子忠于原版）：
  · backbone 不用官方写死的 LLaMA/GPT2/BERT，改用本项目 load_backbone 的 Qwen peft_model
    （inputs_embeds 接口已被 reprogram_backbone 验证可用）
  · 输入由四路剖面在变量维展平成 (B, K, 34) 标量多变量序列
  · 输出只取温跃层深度 (B, H)，用与其它 baseline 一致的三层 MLP head

参考：https://github.com/KimMeen/Time-LLM  models/TimeLLM.py, layers/Embed.py
"""

import math

import torch
import torch.nn as nn

from models.seq_baselines import _build_head  # 与其它 baseline 对齐的三层 MLP 回归头


# ── 官方 layers/Embed.py 移植 ─────────────────────────────────────────────────

class ReplicationPad1d(nn.Module):
    """在序列末尾复制最后一帧 padding 个时间步（官方实现）。"""

    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        # x: (B, C, L) → (B, C, L + padding[-1])
        replicate = x[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        return torch.cat([x, replicate], dim=-1)


class TokenEmbedding(nn.Module):
    """Conv1d(kernel=3) patch → d_model（官方实现）。"""

    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=padding, padding_mode='circular', bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: (B, L, c_in) → (B, L, d_model)
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class PatchEmbedding(nn.Module):
    """时间轴 patch embedding（官方实现）。

    forward: x (B, n_vars, L) → (out (B*n_vars, num_patch, d_model), n_vars)
    """

    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)            # (B*n_vars, num_patch, d_model)
        return self.dropout(x), n_vars


# ── 官方 models/TimeLLM.py ReprogrammingLayer 移植 ────────────────────────────

class ReprogrammingLayer(nn.Module):
    """patch embedding（target）cross-attn 对齐到 LLM source embedding。

    与本项目 reprogram_backbone.ReprogrammingLayer 思路一致，但保持官方签名
    forward(target, source, value)，d_llm 投影由 out_proj 完成。
    """

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_llm,   d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm,   d_keys * n_heads)
        self.out_projection   = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _    = source_embedding.shape
        H       = self.n_heads

        Q = self.query_projection(target_embedding).view(B, L, H, -1)
        K = self.key_projection(source_embedding).view(S, H, -1)
        V = self.value_projection(value_embedding).view(S, H, -1)

        scale = 1.0 / math.sqrt(Q.shape[-1])
        scores = torch.einsum("blhe,she->bhls", Q, K)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", attn, V)        # (B, L, H, e)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)                       # (B, L, d_llm)


# ── Time-LLM forecaster（适配本项目接口）──────────────────────────────────────

def _get_embed_weight(model):
    """在模型层级中查找 embed_tokens 权重（与 reprogram_backbone 一致）。"""
    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and hasattr(module, "weight"):
            return module.weight
    raise AttributeError("embed_tokens not found in model")


# 官方 PaP 风格 prompt 模板：数据集/任务描述 + 自动统计量占位（统计量在 forward 拼入）
_PROMPT_PREFIX = (
    "Dataset: ocean CTD thermocline observations sampled by an AUV. "
    "Task: forecast the future thermocline center depth (meters) over the next "
    "{pred_len} steps from the past {seq_len} steps. "
    "Input statistics: "
)


class TimeLLMForecaster(nn.Module):
    """官方 Time-LLM 架构 + Qwen3.5-2B(LoRA) backbone，输出温跃层深度 (B, H)。

    输入与其它 baseline 对齐：四路原始剖面，内部归一化后在变量维展平。
      depths:  (B, K, 1)
      T_profs: (B, K, N)
      S_profs: (B, K, N)
      context: (B, 3)   [season_norm, doy_sin, doy_cos]
    """

    def __init__(self, peft_model, tokenizer, cfg):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = peft_model.model
        self.llm_dim = peft_model.config.hidden_size

        self.K = cfg.TSF_K
        self.H = cfg.TSF_H
        self.N = cfg.SAMPLE_WINDOW_SIZE
        self.patch_len = cfg.TIMELLM_PATCH_LEN
        self.stride    = cfg.TIMELLM_STRIDE
        self.d_ff      = cfg.TIMELLM_D_FF
        self.num_tokens = cfg.TIMELLM_N_PROTOTYPES

        # depth 归一化常量（与 iTransformer/Reprogram 一致）
        self.register_buffer("depth_min", torch.tensor(float(cfg.TSF_DEPTH_MIN)))
        self.register_buffer("depth_max", torch.tensor(float(cfg.TSF_DEPTH_MAX)))

        # text prototypes：直接取词表 embedding 切片作可学习 prototype。
        # 官方 Time-LLM 用 Linear(vocab→num_tokens) mapping layer，但 Qwen vocab≈151k
        # 会使该层达 ~1.5 亿参数（LLaMA vocab 32k 才适用）；改用词表切片与本工作主方法
        # reprogram_backbone 完全一致（embed_w[100:100+N]），参数降到 num_tokens×llm_dim，
        # 且 prototype 来源与主方法对齐，架构对照更干净。
        with torch.no_grad():
            embed_w0 = _get_embed_weight(self.backbone)
            proto = embed_w0[100 : 100 + self.num_tokens].detach().float().clone()
        self.prototypes = nn.Parameter(proto)   # (num_tokens, llm_dim)

        # 时间轴 patch embedding + reprogramming（float32，送 backbone 前 cast）
        self.patch_embedding = PatchEmbedding(
            cfg.TIMELLM_D_MODEL, self.patch_len, self.stride, cfg.TIMELLM_DROPOUT,
        )
        self.reprogramming_layer = ReprogrammingLayer(
            cfg.TIMELLM_D_MODEL, cfg.TIMELLM_N_HEADS,
            d_llm=self.llm_dim, attention_dropout=cfg.TIMELLM_DROPOUT,
        )
        # patch 输出投到 d_ff（官方 enc_out → d_ff 投影前的 reprogram 直接出 llm_dim，
        # 这里 reprogram 已出 llm_dim；再用 Linear 压到 d_ff 供 head 展平）
        self.out_proj = nn.Linear(self.llm_dim, self.d_ff)

        # num_patch 数（与 PatchEmbedding 的 unfold 对齐）
        self.num_patch = (self.K + self.stride - self.patch_len) // self.stride + 1

        # 变量数：depth(1) + T(N) + S(N) + ctx(3)
        self.n_vars = 1 + self.N + self.N + 3
        nf = self.d_ff * self.num_patch
        # 每个变量产出 nf 维表示，n_vars 个变量拼接后过统一 MLP head → (B, H)
        self.head = _build_head(self.n_vars * nf, self.H)

    def _normalize(self, depths, T_profs, S_profs):
        d_range  = self.depth_max - self.depth_min
        depths_n = (depths - self.depth_min) / d_range * 2.0 - 1.0
        T_n = (T_profs - T_profs.mean(2, keepdim=True)) / (T_profs.std(2, keepdim=True) + 1e-6)
        S_n = (S_profs - S_profs.mean(2, keepdim=True)) / (S_profs.std(2, keepdim=True) + 1e-6)
        return depths_n, T_n, S_n

    def _build_prompt_embeds(self, x_flat, device, llm_dtype):
        """官方 PaP：逐样本算 min/max/median/lags 统计量 → 文本 → embed。

        x_flat: (B, K, n_vars) 已归一化展平输入；返回 (B, L_p, llm_dim)。
        """
        B = x_flat.shape[0]
        embed_w = _get_embed_weight(self.backbone)

        mins = x_flat.min(dim=1).values.mean(dim=1)        # (B,) 各变量均值再聚合，控制 prompt 长度
        maxs = x_flat.max(dim=1).values.mean(dim=1)
        meds = x_flat.median(dim=1).values.mean(dim=1)
        # top-1 lag（自相关最大滞后）：对每样本聚合序列求差分趋势作为简化 lag 描述
        seq = x_flat.mean(dim=2)                            # (B, K)
        trend = (seq[:, -1] - seq[:, 0])                    # (B,)

        prefix = _PROMPT_PREFIX.format(pred_len=self.H, seq_len=self.K)
        texts = [
            prefix + (f"min {mins[i]:.2f}, max {maxs[i]:.2f}, median {meds[i]:.2f}, "
                      f"overall trend {trend[i]:+.2f}. "
                      f"Predict the future thermocline center depth.")
            for i in range(B)
        ]
        tok = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128, add_special_tokens=False,
        )
        ids  = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        with torch.no_grad():
            prompt_embeds = torch.nn.functional.embedding(ids, embed_w).to(llm_dtype).detach()
        return prompt_embeds, mask

    def forward(self, depths, T_profs, S_profs, context):
        B = depths.size(0)
        device = depths.device

        depths_n, T_n, S_n = self._normalize(depths, T_profs, S_profs)
        ctx_exp = context.unsqueeze(1).expand(B, self.K, 3)          # (B, K, 3)
        x = torch.cat([depths_n, T_n, S_n, ctx_exp], dim=2)          # (B, K, n_vars)

        llm_dtype = next(self.backbone.parameters()).dtype

        # ── source embedding：词表切片 prototype（float32），归一化后做 cross-attn key/value ──
        source_embeddings = torch.nn.functional.normalize(self.prototypes, dim=-1)  # (num_tokens, llm_dim)

        # ── 时间轴 patch + reprogramming（float32）──
        x_in = x.permute(0, 2, 1)                                     # (B, n_vars, K)
        enc_out, n_vars = self.patch_embedding(x_in.float())          # (B*n_vars, num_patch, d_model)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)  # (B*n_vars, num_patch, llm_dim)

        # ── PaP prompt 前缀拼接，过 backbone ──
        prompt_embeds, prompt_mask = self._build_prompt_embeds(x, device, llm_dtype)  # (B, L_p, llm_dim)
        L_p = prompt_embeds.shape[1]
        # enc_out 按变量展开回 (B, n_vars*num_patch, llm_dim)，与每样本 prompt 拼接
        enc_out = enc_out.reshape(B, n_vars * self.num_patch, self.llm_dim).to(llm_dtype)
        numeric_mask = torch.ones(B, enc_out.shape[1], dtype=torch.long, device=device)

        inputs_embeds = torch.cat([prompt_embeds, enc_out], dim=1)    # (B, L_p + n_vars*num_patch, llm_dim)
        attn_mask     = torch.cat([prompt_mask, numeric_mask], dim=1)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]                            # (B, L_total, llm_dim)

        # 只取数值（patch）部分，按变量 reshape，压到 d_ff，展平过 head
        numeric_hidden = hidden[:, L_p:, :].float()                  # (B, n_vars*num_patch, llm_dim)
        numeric_hidden = self.out_proj(numeric_hidden)              # (B, n_vars*num_patch, d_ff)
        pooled = numeric_hidden.reshape(B, -1)                      # (B, n_vars*num_patch*d_ff)
        return self.head(pooled)                                    # (B, H)


def load_timellm(cfg):
    """加载 Qwen(LoRA) backbone 并构造 TimeLLMForecaster。"""
    from models.llm_backbone import load_backbone

    peft_model, tokenizer = load_backbone(cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = TimeLLMForecaster(peft_model, tokenizer, cfg)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TimeLLMForecaster loaded. Params total: {n_total:,}  trainable: {n_train:,}")
    return model
