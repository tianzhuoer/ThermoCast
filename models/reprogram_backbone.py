"""
ThermoForecasterReprogram: reprogramming-based thermocline forecaster.

Each timestep t produces 3 tokens (depth, T-profile, S-profile) that are
aligned to the LLM embedding space via per-modality ReprogrammingLayers.
Season context is injected via a text prompt with a [THERMO] placeholder:
  the placeholder token is replaced by ctx_token (reprogram output of context).

Architecture:
  context  (B,3)   ────→ Linear → d_model → ReprogrammingLayer → ctx_token (B,1,llm_dim)
  depths   (B,K,1) ─┐
  T_profs  (B,K,N) ──→ Linear → d_model → ReprogrammingLayer → (B,3K,llm_dim)
  S_profs  (B,K,N) ─┘

  prompt text → tokenize → embed_tokens → replace [THERMO] pos with ctx_token
                                                     ↓
                          cat([prompt_embeds, numeric_tokens]) → (B, L_p+3K, llm_dim)
                                                     ↓
                          Qwen backbone (LoRA, inputs_embeds)
                                                     ↓
                          last+mean pool over numeric tokens only → regression head → (B, H)
"""

import torch
import torch.nn as nn

THERMO_PLACEHOLDER = "[THERMO]"

SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}

# 各季节温跃层典型深度范围（与 configs.shared.SEASON_TYPICAL_DEPTHS 一致），
# 作为 [THERMO] 占位符之外的文字背景，动态 prompt 仍由 ctx_token 注入精确范围。
SEASON_RANGE = {0: "65-125 m", 1: "45-105 m", 2: "25-85 m", 3: "45-125 m"}


def _trend_word(delta: float, thr: float) -> str:
    """把首尾差值映射为趋势词。thr 为判定为 steady 的死区半宽。"""
    if delta < -thr:
        return "dropped"
    if delta > thr:
        return "rose"
    return "steady"


def _grad_word(t_trend: str, s_trend: str) -> str:
    """T/S 趋势 → 密度跃层强弱的定性物理描述（把数值相关性翻成语言）。

    低温高盐→密度增→梯度锐化；高温低盐→密度减→梯度模糊。
    """
    if t_trend == "dropped" and s_trend == "rose":
        return "strengthening"
    if t_trend == "rose" and s_trend == "dropped":
        return "weakening"
    if t_trend == "steady" or s_trend == "steady":
        return "stable"
    return "shifting"


def build_dynamic_prompt(season_id: int,
                         d_mean: float, d_min: float, d_max: float,
                         t_mean: float, dT: float,
                         s_mean: float, dS: float,
                         ts_corr: float) -> str:
    """根据当前窗口统计量构造动态文本 prompt（含 [THERMO] 占位符）。

    数值压成统计量后用自然语言描述，让一部分输入落在 LLM 熟悉的文本分布里，
    激活预训练语义；精确 CTD 序列仍由 reprogram numeric token 处理。
    T-S 相关性写成「定量 r + 定性物理词」两层，从当前窗口算出，非静态先验。
    """
    season   = SEASON_NAMES.get(int(season_id), "Unknown")
    t_trend  = _trend_word(dT, 0.3)
    s_trend  = _trend_word(dS, 0.1)
    grad     = _grad_word(t_trend, s_trend)
    return (
        f"You are an oceanographer predicting thermocline center depth. "
        f"Season: {season}. Typical range: {THERMO_PLACEHOLDER}. "
        f"Last steps: AUV depth mean {d_mean:.0f} m (range {d_min:.0f} to {d_max:.0f} m). "
        f"Temperature {t_trend} {abs(dT):.1f} C (mean {t_mean:.1f} C); "
        f"salinity {s_trend} {abs(dS):.1f} PSU (mean {s_mean:.1f} PSU). "
        f"T-S correlation {ts_corr:+.1f}; density gradient {grad}. "
        f"Predict the future thermocline center depth."
    )

# context 向量维度：[season_norm, doy_sin, doy_cos, depth_mean_n, depth_range_n, T_mean_n, T_std_n, S_mean_n, S_std_n]
# depth_mean_n / depth_range_n 归一化到 [-1,1]；T/S 统计量 z-score 后 clip 到 [-3,3]
CTX_DIM = 9

# prompt 包含季节 + AUV 观测的关键数值统计，[THERMO] 被 ctx_token 替换
# 统计信息由调用方在 forward 前格式化进 prompt_text，此处为模板
SEASON_PROMPTS = {
    0: ("You are predicting ocean thermocline center depth. "
        "Season: Winter. AUV depth range: " + THERMO_PLACEHOLDER + ". "
        "The numerical observations follow. Predict the thermocline center depth."),
    1: ("You are predicting ocean thermocline center depth. "
        "Season: Spring. AUV depth range: " + THERMO_PLACEHOLDER + ". "
        "The numerical observations follow. Predict the thermocline center depth."),
    2: ("You are predicting ocean thermocline center depth. "
        "Season: Summer. AUV depth range: " + THERMO_PLACEHOLDER + ". "
        "The numerical observations follow. Predict the thermocline center depth."),
    3: ("You are predicting ocean thermocline center depth. "
        "Season: Autumn. AUV depth range: " + THERMO_PLACEHOLDER + ". "
        "The numerical observations follow. Predict the thermocline center depth."),
}


def _get_embed_weight(model):
    """Find embed_tokens weight anywhere in the model hierarchy."""
    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and hasattr(module, "weight"):
            return module.weight
    raise AttributeError("embed_tokens not found in model")


class ReprogrammingLayer(nn.Module):
    """
    Cross-attention: patch embeddings (d_model) → LLM prototype space (d_llm).
    Operates in float32; caller is responsible for dtype casting around LLM.
    """

    def __init__(self, d_model: int, n_heads: int, d_llm: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_k = d_k
        self.scale = d_k ** -0.5

        self.q_proj   = nn.Linear(d_model, d_k * n_heads)
        self.k_proj   = nn.Linear(d_llm,   d_k * n_heads)
        self.v_proj   = nn.Linear(d_llm,   d_k * n_heads)
        self.out_proj = nn.Linear(d_k * n_heads, d_llm)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # x:      (B, L, d_model)   float32
        # source: (S, d_llm)        float32
        # return: (B, L, d_llm)     float32
        B, L, _ = x.shape
        S, _    = source.shape
        h, dk   = self.n_heads, self.d_k

        Q = self.q_proj(x).view(B, L, h, dk)
        K = self.k_proj(source).view(S, h, dk)
        V = self.v_proj(source).view(S, h, dk)

        scores = torch.einsum("blhd,shd->bhls", Q, K) * self.scale
        attn   = self.dropout(torch.softmax(scores, dim=-1))
        out    = torch.einsum("bhls,shd->blhd", attn, V).reshape(B, L, h * dk)
        return self.out_proj(out)


def _pool_numeric(hidden: torch.Tensor, L_p: int, K: int) -> torch.Tensor:
    """Pool over numeric tokens only (last 3K positions). → (B, 2*llm_dim)"""
    numeric = hidden[:, L_p:, :].float()   # (B, 3K, llm_dim)
    mean_p  = numeric.mean(dim=1)          # (B, llm_dim)
    last_p  = numeric[:, -1, :]            # (B, llm_dim)
    return torch.cat([last_p, mean_p], dim=-1)


class ThermoForecasterReprogram(nn.Module):
    def __init__(
        self,
        peft_model,
        tokenizer,
        H: int,
        K: int,
        N: int = 15,
        d_model: int = 64,
        n_heads: int = 4,
        n_prototypes: int = 64,
        depth_min: float = -150.0,
        depth_max: float = -25.0,
        dropout: float = 0.1,
        use_reprogram: bool = True,
        use_backbone: bool = True,
        use_dynamic_prompt: bool = False,
    ):
        super().__init__()
        self.H = H
        self.K = K
        self.use_reprogram = use_reprogram
        self.use_backbone = use_backbone
        self.use_dynamic_prompt = use_dynamic_prompt
        self.tokenizer = tokenizer
        llm_dim = peft_model.config.hidden_size
        # 动态 prompt 模式下 batch 内 prompt 长度不一，需要的 pad/[THERMO] 上限
        self._dyn_pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 预计算 4 个季节 prompt 的 token ids 及 [THERMO] 占位符位置
        # _prompt_ids:  dict[int, Tensor(L_p,)]   各季节等长（断言保证）
        # _thermo_pos:  dict[int, int]             占位符在 token 序列中的索引
        self._prompt_ids:  dict[int, torch.Tensor] = {}
        self._thermo_pos:  dict[int, int] = {}

        # 把 prompt 在 [THERMO] 处拆成前后两段分别 tokenize，避免 BPE 上下文依赖导致匹配失败
        # pos = 前段 token 数，即 [THERMO] 在拼接后序列中的位置
        for sid, prompt_text in SEASON_PROMPTS.items():
            assert THERMO_PLACEHOLDER in prompt_text, f"[THERMO] not in prompt for season {sid}"
            pre, post = prompt_text.split(THERMO_PLACEHOLDER, maxsplit=1)

            pre_ids  = tokenizer(pre,  return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

            # 用一个 pad_id 占位 [THERMO]，forward 里会被替换为 ctx_token embedding
            placeholder = torch.tensor(
                [tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0],
                dtype=torch.long,
            )
            ids_trimmed = torch.cat([pre_ids, placeholder, post_ids])  # (L_p_sid,)
            self._prompt_ids[sid] = ids_trimmed
            self._thermo_pos[sid] = pre_ids.size(0)

        # 各季节 prompt pad 到同一长度，pad 位用 attention mask 屏蔽
        # _prompt_mask: dict[int, Tensor(L_p,)]  1=真实token, 0=pad
        L_p = max(ids.size(0) for ids in self._prompt_ids.values())
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._prompt_len = L_p
        for sid in list(self._prompt_ids.keys()):
            ids  = self._prompt_ids[sid]
            diff = L_p - ids.size(0)
            if diff > 0:
                # pad 在末尾（[THERMO] 在句中，位置不受影响）
                padding = torch.full((diff,), pad_id, dtype=torch.long)
                self._prompt_ids[sid] = torch.cat([ids, padding])
        # _prompt_mask 在 forward 里按实际长度动态生成，避免存储冗余

        # per-modality patch projections (float32)
        self.depth_proj = nn.Linear(1, d_model)
        self.T_proj     = nn.Linear(N, d_model)
        self.S_proj     = nn.Linear(N, d_model)
        # context: [season_norm, doy_sin, doy_cos, depth_mean_n, depth_range_n,
        #           T_mean_n, T_std_n, S_mean_n, S_std_n]  → CTX_DIM=9
        self.ctx_proj   = nn.Linear(CTX_DIM, d_model)

        if use_reprogram:
            # per-modality reprogramming layers (cross-attn with prototypes)
            self.reprogram_depth = ReprogrammingLayer(d_model, n_heads, llm_dim, dropout)
            self.reprogram_T     = ReprogrammingLayer(d_model, n_heads, llm_dim, dropout)
            self.reprogram_S     = ReprogrammingLayer(d_model, n_heads, llm_dim, dropout)
            self.reprogram_ctx   = ReprogrammingLayer(d_model, n_heads, llm_dim, dropout)
            with torch.no_grad():
                embed_w = _get_embed_weight(peft_model)
                proto   = embed_w[100 : 100 + n_prototypes].detach().float().clone()
            self.prototypes = nn.Parameter(proto)  # (n_prototypes, llm_dim)
        else:
            # 消融：用简单 Linear 替换 ReprogrammingLayer，去掉 prototype cross-attn
            self.linear_depth = nn.Linear(d_model, llm_dim)
            self.linear_T     = nn.Linear(d_model, llm_dim)
            self.linear_S     = nn.Linear(d_model, llm_dim)
            self.linear_ctx   = nn.Linear(d_model, llm_dim)

        # depth normalization constants (registered as buffers → device-portable)
        self.register_buffer("depth_min", torch.tensor(depth_min))
        self.register_buffer("depth_max", torch.tensor(depth_max))

        # normalize reprogrammed embeddings to LLM distribution before backbone
        self.embed_norm = nn.LayerNorm(llm_dim)

        # LoRA-adapted LLM backbone
        self.backbone = peft_model.model

        # regression head (float32)
        self.regressor = nn.Sequential(
            nn.LayerNorm(llm_dim * 2),
            nn.Linear(llm_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, H),
        )

    # ------------------------------------------------------------------
    def _normalize_inputs(self, depths, T_profs, S_profs):
        d_range  = self.depth_max - self.depth_min
        depths_n = (depths - self.depth_min) / d_range * 2.0 - 1.0  # → [-1,1]
        T_n = (T_profs - T_profs.mean(2, keepdim=True)) / (T_profs.std(2, keepdim=True) + 1e-6)
        S_n = (S_profs - S_profs.mean(2, keepdim=True)) / (S_profs.std(2, keepdim=True) + 1e-6)
        return depths_n, T_n, S_n

    def _build_dynamic_prompt_ids(self, depths, T_profs, S_profs, season_ids, device):
        """逐样本构造动态文本 prompt → tokenize → batch 内 pad 对齐。

        统计量全部用物理单位（depths 米、T/S 原始值），落在文本分布里。
        返回：
          prompt_ids_padded (B, L_max)  动态文本 token ids，pad 到 batch 最长
          thermo_positions  (B,)        各样本 [THERMO] 占位符位置
          L_max             int         本 batch 最大 prompt 长度
        """
        B = depths.size(0)
        # 每步剖面均值 → 长度 K 的 T/S 序列；与 depth 一起算窗口统计量
        d_seq = depths[:, :, 0]                      # (B, K)
        T_seq = T_profs.mean(dim=2)                  # (B, K)
        S_seq = S_profs.mean(dim=2)                  # (B, K)

        d_mean = d_seq.mean(dim=1)                   # (B,)
        d_min  = d_seq.min(dim=1).values
        d_max  = d_seq.max(dim=1).values
        t_mean = T_seq.mean(dim=1)
        s_mean = S_seq.mean(dim=1)
        dT     = T_seq[:, -1] - T_seq[:, 0]          # 窗口首尾温差
        dS     = S_seq[:, -1] - S_seq[:, 0]

        # T-S Pearson 相关（沿时间维），nan→0
        T_c = T_seq - T_seq.mean(dim=1, keepdim=True)
        S_c = S_seq - S_seq.mean(dim=1, keepdim=True)
        cov = (T_c * S_c).mean(dim=1)
        std = T_c.std(dim=1) * S_c.std(dim=1) + 1e-6
        corr = torch.nan_to_num(cov / std, nan=0.0).clamp(-1.0, 1.0)  # (B,)

        pad_id = self._dyn_pad_id
        ids_list, thermo_list = [], []
        for i in range(B):
            text = build_dynamic_prompt(
                int(season_ids[i]),
                float(d_mean[i]), float(d_min[i]), float(d_max[i]),
                float(t_mean[i]), float(dT[i]),
                float(s_mean[i]), float(dS[i]),
                float(corr[i]),
            )
            pre, post = text.split(THERMO_PLACEHOLDER, maxsplit=1)
            pre_ids  = self.tokenizer(pre,  return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            placeholder = torch.tensor([pad_id], dtype=torch.long)
            ids = torch.cat([pre_ids, placeholder, post_ids])
            ids_list.append(ids)
            thermo_list.append(pre_ids.size(0))

        L_max = max(ids.size(0) for ids in ids_list)
        prompt_ids_padded = torch.full((B, L_max), pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            prompt_ids_padded[i, : ids.size(0)] = ids
        prompt_ids_padded = prompt_ids_padded.to(device)
        thermo_positions  = torch.tensor(thermo_list, dtype=torch.long, device=device)
        return prompt_ids_padded, thermo_positions, L_max

    def forward(self, depths, T_profs, S_profs, context, season_ids):
        """
        depths:     (B, K, 1)   float32   AUV depth
        T_profs:    (B, K, N)   float32   temperature profile
        S_profs:    (B, K, N)   float32   salinity profile
        context:    (B, 9)      float32   [season_norm, doy_sin, doy_cos,
                                           depth_mean_n, depth_range_n,
                                           T_mean_n, T_std_n, S_mean_n, S_std_n]
        season_ids: (B,)        int64     season index 0-3
        """
        B      = depths.size(0)
        device = depths.device

        depths_n, T_n, S_n = self._normalize_inputs(depths, T_profs, S_profs)

        if self.use_reprogram:
            src   = torch.nn.functional.normalize(self.prototypes, dim=-1)
            d_r   = self.reprogram_depth(self.depth_proj(depths_n), src)
            T_r   = self.reprogram_T(self.T_proj(T_n),               src)
            S_r   = self.reprogram_S(self.S_proj(S_n),               src)
            ctx_r = self.reprogram_ctx(self.ctx_proj(context).unsqueeze(1), src)
        else:
            d_r   = self.linear_depth(self.depth_proj(depths_n))               # (B, K, llm_dim)
            T_r   = self.linear_T(self.T_proj(T_n))                            # (B, K, llm_dim)
            S_r   = self.linear_S(self.S_proj(S_n))                            # (B, K, llm_dim)
            ctx_r = self.linear_ctx(self.ctx_proj(context)).unsqueeze(1)       # (B, 1, llm_dim)

        # numeric tokens: interleave [d_0,T_0,S_0, …] → (B, 3K, llm_dim)
        numeric_tokens = torch.stack([d_r, T_r, S_r], dim=2).reshape(B, 3 * self.K, -1)
        numeric_tokens = self.embed_norm(numeric_tokens)

        # 诊断消融：完全跳过 backbone（含 prompt 路径），reprogram 输出直接池化送 regressor。
        # 用于验证 LLM backbone（无论预训练/随机）在当前架构下是否真的参与预测。
        if not self.use_backbone:
            num = numeric_tokens.float()              # (B, 3K, llm_dim)
            mean_p = num.mean(dim=1)                  # (B, llm_dim)
            last_p = num[:, -1, :]                    # (B, llm_dim)
            pooled = torch.cat([last_p, mean_p], dim=-1)
            return self.regressor(pooled)             # (B, H)

        llm_dtype  = next(self.backbone.parameters()).dtype
        embed_w    = _get_embed_weight(self.backbone)  # (vocab, llm_dim)

        # 构建 prompt embeddings。动态模式逐样本生成统计量文本并 tokenize；
        # 静态模式查表 4 个季节定长模板（旧行为，实验对照用）。
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if self.use_dynamic_prompt:
            prompt_ids_padded, thermo_positions, L_p = self._build_dynamic_prompt_ids(
                depths, T_profs, S_profs, season_ids, device
            )
        else:
            L_p = self._prompt_len
            prompt_ids_padded = torch.stack(
                [self._prompt_ids[int(sid)].to(device) for sid in season_ids], dim=0
            )  # (B, L_p)
            thermo_positions = torch.tensor(
                [self._thermo_pos[int(sid)] for sid in season_ids],
                dtype=torch.long, device=device,
            )  # (B,)

        with torch.no_grad():
            prompt_embeds_base = torch.nn.functional.embedding(
                prompt_ids_padded, embed_w
            ).to(llm_dtype).detach()                               # (B, L_p, llm_dim)，无梯度

        # 把每个样本 prompt 里的 [THERMO] 位置替换为 ctx_token
        # 用 torch.where + bool mask，保证 ctx_r 梯度路径完整
        # thermo_mask: (B, L_p, 1) → broadcast 到 (B, L_p, llm_dim)，True 处取 ctx_token
        pos_range = torch.arange(L_p, device=device).unsqueeze(0)             # (1, L_p)
        thermo_mask = (pos_range == thermo_positions.unsqueeze(1))             # (B, L_p) bool
        thermo_mask = thermo_mask.unsqueeze(2).expand_as(prompt_embeds_base)  # (B, L_p, llm_dim)
        ctx_token = ctx_r[:, 0:1, :].to(llm_dtype).expand(B, L_p, -1)        # (B, L_p, llm_dim)
        # torch.where：True 位置取 ctx_token（有梯度），False 位置取静态 prompt embeds
        prompt_embeds = torch.where(thermo_mask, ctx_token, prompt_embeds_base)

        numeric_tokens_llm = numeric_tokens.to(llm_dtype)
        # prompt_mask: pad 位置为 0，真实 token 位置为 1
        prompt_mask  = (prompt_ids_padded != pad_id).long()        # (B, L_p)
        # [THERMO] 位置已被 ctx_token 覆盖，强制有效
        prompt_mask  = torch.where(thermo_mask[:, :, 0],
                                   torch.ones_like(prompt_mask), prompt_mask)
        numeric_mask = torch.ones(B, 3 * self.K, dtype=torch.long, device=device)

        inputs_embeds = torch.cat([prompt_embeds, numeric_tokens_llm], dim=1)  # (B, L_p+3K, llm_dim)
        attn_mask     = torch.cat([prompt_mask,   numeric_mask],        dim=1)  # (B, L_p+3K)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        hidden = outputs.hidden_states[-1]                    # (B, L_p+3K, llm_dim)
        pooled = _pool_numeric(hidden, L_p, self.K)           # (B, 2*llm_dim)
        return self.regressor(pooled)                         # (B, H)
