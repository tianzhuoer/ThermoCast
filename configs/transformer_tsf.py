from .shared import *

# ── 模型结构 ──────────────────────────────────────────────────────────────────
# 原始剖面输入维度：1(depth) + N(T_prof) + N(S_prof)，不含季节/doy
RAW_INPUT_SIZE = 1 + SAMPLE_WINDOW_SIZE + SAMPLE_WINDOW_SIZE      # = 31

TF_D_MODEL    = 32    # 小容量，无法充分利用 34 维原始剖面
TF_N_HEADS    = 2     # 每头 16 维
TF_NUM_LAYERS = 2     # Encoder 层数
TF_FFN_DIM    = 64    # FFN 中间维度（2×d_model）
TF_DROPOUT    = 0.1

# ── 训练时噪声增强（与 Reprogram 保持一致）
DEPTH_NOISE_STD = 1.0
FEAT_NOISE_STD  = 0.01

# ── 训练参数 ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LR         = 1e-4

# ── 检查点 ────────────────────────────────────────────────────────────────────
TSF_CKPT_DIR = "checkpoints/transformer"
TSF_BEST_DIR = "checkpoints/transformer/best"

HUBER_DELTA = 0.5    # 覆盖 shared 默认值；输入换为原始剖面后误差分布更宽，需更大 delta

# ── SwanLab ───────────────────────────────────────────────────────────────────
SWANLAB_EXPERIMENT = f"transformer-tsf-{MAX_SAMPLES_PER_EPOCH//1000}k"
