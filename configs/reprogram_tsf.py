from .shared import *

# ── 模型路径（Qwen3.5-2B）─────────────────────────────────────────────────────
MODEL_PATH = os.getenv("THERMOCAST_QWEN_PATH", "Qwen/Qwen3.5-2B")

# ── Reprogramming 超参数 ────────────────────────────────────────────────────────
REPROGRAM_D_MODEL      = 64
REPROGRAM_N_HEADS      = 4
REPROGRAM_N_PROTOTYPES = 128   # 64→128：更多 prototype，参数量极小（128×llm_dim），无显存压力
REPROGRAM_DROPOUT      = 0.1

# ── 训练超参数 ─────────────────────────────────────────────────────────────────
BATCH_SIZE       = 8   # 加入 FFN LoRA 后显存增加，减半 batch
GRAD_ACCUM_STEPS = 8   # ×2 补偿 batch 减半，等效 batch=64 不变

# ── LoRA 配置 ─────────────────────────────────────────────────────────────────
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGET  = ["k_proj", "q_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]   # 加入 FFN 层，让 backbone 能调整语义响应

# ── 学习率 ────────────────────────────────────────────────────────────────────
# LR_REPROGRAM / LR_HEAD 同时用于 --mode lora 与 --mode head（冻结 LLM 消融与主实验保持一致）
LR_LORA      = 1e-5
LR_REPROGRAM = 8e-5   # 适当提高：reprogram 层需要更快适配 LLM embedding 空间
LR_HEAD      = 8e-5
WARMUP_RATIO = 0.08   # 3%→8%：延长 warmup，让 LR 在峰值区间停留更久再衰减
GRAD_CLIP_MAX_NORM = 0.5

# ── 训练过程噪声增强 ──────────────────────────────────────────────────────────
DEPTH_NOISE_STD = 1.5    # 1.0→1.5：更强深度扰动，防止过拟合到训练轨迹
FEAT_NOISE_STD  = 0.02   # 0.01→0.02：更强温盐扰动

# ── 检查点 ─────────────────────────────────────────────────────────────────────
# 根目录；具体 variant 子目录（如 reprogram/lora-16k、reprogram/head-randinit-16k）
# 由 train_reprogram_TSF.py 按 variant_tag 动态拼接，避免不同模式互相覆盖。
TSF_CKPT_ROOT = "checkpoints/reprogram"
TSF_CKPT_DIR  = TSF_CKPT_ROOT   # 默认值；脚本会按 variant 覆盖为子目录
TSF_LORA_BEST = f"{TSF_CKPT_DIR}/lora_best"
TSF_HEAD_BEST = f"{TSF_CKPT_DIR}/head_best.pt"

# ── SwanLab ────────────────────────────────────────────────────────────────────
SWANLAB_EXPERIMENT = f"qwen3.5-2b-reprogram-tsf-{MAX_SAMPLES_PER_EPOCH//1000}k"
