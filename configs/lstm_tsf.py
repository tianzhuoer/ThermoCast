from .shared import *

# ── 原始剖面输入维度：1(depth) + N(T_prof) + N(S_prof) + 3(season/doy) ────────
# N = SAMPLE_WINDOW_SIZE = 15，与 Reprogram 输入对齐
RAW_INPUT_SIZE = 1 + SAMPLE_WINDOW_SIZE + SAMPLE_WINDOW_SIZE + 3  # = 34

# ── 训练时噪声增强（与 Reprogram 保持一致）────────────────────────────────────
DEPTH_NOISE_STD = 1.0    # 输入深度噪声（米）
FEAT_NOISE_STD  = 0.01   # 温盐观测噪声

# ── 模型结构 ──────────────────────────────────────────────────────────────────
LSTM_HIDDEN_SIZE = 128   # 隐层维度
LSTM_NUM_LAYERS  = 2     # 层数
LSTM_DROPOUT     = 0.1   # 层间 dropout（num_layers=1 时自动禁用）

# ── 训练参数 ──────────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
LR          = 3e-4   # 全参数单一学习率，随机初始化模型无需分层 LR
HUBER_DELTA = 0.5    # 覆盖 shared 默认值；输入换为原始剖面后误差分布更宽，需更大 delta

# ── 检查点 ────────────────────────────────────────────────────────────────────
TSF_CKPT_DIR = "checkpoints/lstm"
TSF_BEST_DIR = "checkpoints/lstm/best"

# ── SwanLab ───────────────────────────────────────────────────────────────────
SWANLAB_EXPERIMENT = f"lstm-tsf-{MAX_SAMPLES_PER_EPOCH//1000}k"
