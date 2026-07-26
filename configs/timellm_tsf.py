from .shared import *

# ══════════════════════════════════════════════════════════════════════════════
# Time-LLM（ICLR 2024）baseline 配置 —— 架构对照
#
# 与主方法 Reprogram-TSF 同 backbone（Qwen3.5-2B）、同微调策略（LoRA），仅把架构
# 换成官方 Time-LLM 那套（时间轴 patch + text prototypes mapping + Prompt-as-Prefix）。
# 这样对比变量只剩「架构设计」，可干净归因到本工作的 patch 轴迁移与季节 prompt 创新。
# ══════════════════════════════════════════════════════════════════════════════

# ── 模型路径（与 reprogram_tsf 一致：Qwen3.5-2B）─────────────────────────────────
MODEL_PATH = os.getenv("THERMOCAST_QWEN_PATH", "Qwen/Qwen3.5-2B")

# ── LoRA 配置（与 reprogram_tsf 对齐；LORA_R 受显存限制 ≤ 8）──────────────────────
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGET  = ["k_proj", "q_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# ── Time-LLM 架构超参（官方默认量级）────────────────────────────────────────────
TIMELLM_PATCH_LEN    = 8     # 时间轴 patch 长度
TIMELLM_STRIDE       = 4     # patch 步长（K=32 → num_patch=(32+4-8)/4+1=8）
TIMELLM_D_MODEL      = 32    # patch embedding 维度
TIMELLM_N_HEADS      = 8     # ReprogrammingLayer 注意力头数
TIMELLM_N_PROTOTYPES = 1000  # text prototype 数（取词表 embedding 前 N 个切片，与主方法一致）
                             # 官方用 Linear(vocab→N) mapping，但 Qwen vocab≈151k 会致 ~1.5 亿参数 OOM，
                             # 故改用词表切片（参数仅 N×llm_dim），与 reprogram_backbone 对齐
TIMELLM_D_FF         = 32    # reprogram 输出压缩维度（供 head 展平，控制 head 参数量）
TIMELLM_DROPOUT      = 0.1

# ── 训练超参（参考 reprogram_tsf：含 LLM backbone，显存吃紧，batch 偏小）──────────
BATCH_SIZE       = 8
GRAD_ACCUM_STEPS = 8   # 等效 batch=64

# ── 学习率（分组：LoRA 较低，reprogram/mapping/head 较高，参考 reprogram）─────────
LR_LORA      = 1e-5
LR_REPROGRAM = 8e-5
LR_HEAD      = 8e-5
WARMUP_RATIO = 0.08
GRAD_CLIP_MAX_NORM = 0.5

# ── 训练过程噪声增强（与 iTransformer/Reprogram 一致）────────────────────────────
DEPTH_NOISE_STD = 1.0
FEAT_NOISE_STD  = 0.01

# ── 检查点 ─────────────────────────────────────────────────────────────────────
TSF_CKPT_DIR  = "checkpoints/timellm"
TSF_LORA_BEST = f"{TSF_CKPT_DIR}/lora_best"
TSF_HEAD_BEST = f"{TSF_CKPT_DIR}/head_best.pt"
TSF_BEST_DIR  = f"{TSF_CKPT_DIR}/best"

# ── SwanLab ────────────────────────────────────────────────────────────────────
SWANLAB_EXPERIMENT = f"qwen3.5-2b-timellm-tsf-{MAX_SAMPLES_PER_EPOCH//1000}k"
