from .shared import *

# ── 模型结构 ──────────────────────────────────────────────────────────────────
IT_D_MODEL    = 128   # variate token 嵌入维度（Linear(K → d_model)）
IT_N_HEADS    = 4     # 跨 variate 的注意力头数（每头 32 维）
IT_NUM_LAYERS = 2     # Encoder 层数
IT_FFN_DIM    = 256   # FFN 中间维度（2×d_model）
IT_DROPOUT    = 0.1   # Attention 和 FFN 的 dropout

# ── 训练时噪声增强（与 Reprogram 保持一致）────────────────────────────────────
DEPTH_NOISE_STD = 1.0    # 输入深度噪声（米）
FEAT_NOISE_STD  = 0.01   # 温盐观测噪声

# ── 训练参数 ──────────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
LR          = 1e-5   # 全参数单一学习率

# ── 检查点 ────────────────────────────────────────────────────────────────────
TSF_CKPT_DIR = "checkpoints/itransformer"
TSF_BEST_DIR = "checkpoints/itransformer/best"

# ── SwanLab ───────────────────────────────────────────────────────────────────
SWANLAB_EXPERIMENT = f"itransformer-tsf-{MAX_SAMPLES_PER_EPOCH//1000}k"
