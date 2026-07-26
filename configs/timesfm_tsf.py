from .shared import *

# ── 模型 ────────────────────────────────────────────────────────────────────
TIMESFM_MODEL_ID  = "google/timesfm-2.5-200m-transformers"
TIMESFM_CACHE_DIR = os.getenv("HF_HOME")

# TimesFM 内部按 patch_len=32 分片，输入长度必须是 32 的整数倍
TIMESFM_INPUT_LEN = 32

# ── 评估参数 ──────────────────────────────────────────────────────────────────
# EVAL_H 可超过 TSF_H（超出时自动截断）
EVAL_H          = 20
BATCH_SIZE_EVAL = 64
DEVICE          = "cuda"

# ── 微调参数 ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32    # 每个 GPU 的 batch size
LR_FT_BACKBONE  = 5e-7  # backbone 学习率：极小值防止过拟合，但允许 backbone 向任务分布适配
LR_FT_HEAD      = 3e-5  # input_proj + projection_head 学习率
WEIGHT_DECAY_HEAD = 1e-2  # head 参数的 weight decay，比 backbone 更强，抑制过拟合

# 全参数微调（backbone + head）；冻结 backbone 会导致预训练表示对新信号无意义
FREEZE_BACKBONE = False

# ── 检查点 ─────────────────────────────────────────────────────────────────────
TSF_CKPT_DIR  = "checkpoints/timesfm"
TSF_BEST_DIR  = "checkpoints/timesfm/best"

# ── SwanLab ────────────────────────────────────────────────────────────────────
SWANLAB_EXPERIMENT = "timesfm-ft"
