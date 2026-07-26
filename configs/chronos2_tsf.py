from .shared import *

# ── 模型 ──────────────────────────────────────────────────────────────────────
CHRONOS2_MODEL_ID  = "amazon/chronos-2"
CHRONOS2_CACHE_DIR = os.getenv("HF_HOME")

# ── 评估参数 ──────────────────────────────────────────────────────────────────
EVAL_H          = 20
BATCH_SIZE_EVAL = 64
DEVICE          = "cuda"

# ── 微调参数 ──────────────────────────────────────────────────────────────────
BATCH_SIZE        = 32    # 每个 GPU 的 batch size
LR_FT_BACKBONE    = 5e-7  # backbone 学习率：极小值防止过拟合，但允许 backbone 向任务分布适配
LR_FT_HEAD        = 3e-5  # input_proj + projection_head 学习率
WEIGHT_DECAY_HEAD = 1e-2  # head 参数的 weight decay，比 backbone 更强，抑制过拟合

# 全参数微调（backbone + head）；冻结 backbone 会导致预训练表示对新信号无意义
FREEZE_BACKBONE = False

# ── 检查点 ────────────────────────────────────────────────────────────────────
TSF_CKPT_DIR = "checkpoints/chronos2"
TSF_BEST_DIR = "checkpoints/chronos2/best"

# ── SwanLab ───────────────────────────────────────────────────────────────────
SWANLAB_EXPERIMENT = f"chronos2-ft-{MAX_SAMPLES_PER_EPOCH//1000}k"
