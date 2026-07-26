import os
import sys

# ── 随机种子 ──────────────────────────────────────────────────────────────────
# MANUAL_SEED = random.randint(0, 999999)
MANUAL_SEED = 20058

# ── CTD 环境参数（AUV 采样模拟所需）─────────────────────────────────────────
CTD_FOLDER_PATH    = os.getenv("THERMOCAST_CTD_PATH", "data/CTD")
DEPTH_MIN          = -250
DEPTH_MAX          = 0
SAMPLE_WINDOW_SIZE = 15
STEP_SIZE          = 5

# ── TSF 任务参数 ──────────────────────────────────────────────────────────────
TSF_K = 32   # 输入窗口步数
TSF_H = 5    # 预测地平线步数

# 统一数值输入特征维度（depth + T_grad + avg_T + S_grad + avg_S + season + doy_sin + doy_cos）
# 所有数值模型（TimesFM、Chronos-2、LSTM、Transformer）共用，确保输入信息与 Qwen3 对齐
N_INPUT_FEATURES = 8

# 温跃层归一化边界（米）：实际出现范围比全深度 [-250, 0] 窄，
# 收窄范围可让回归头在更密集的目标分布上训练，梯度信号更清晰。
TSF_DEPTH_MIN = -150
TSF_DEPTH_MAX = -25

# ── 数据集划分 ────────────────────────────────────────────────────────────────
# 按每个 CTD 文件时间轴的 70/20/10 比例划分（保证所有文件均参与各 split）
SPLIT_TRAIN_MOD = set(range(7))
SPLIT_VAL_MOD   = {7, 8}
SPLIT_TEST_MOD  = {9}

# ── 季节信息 ──────────────────────────────────────────────────────────────────
SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
SEASON_TYPICAL_DEPTHS = {
    0: "65–125 m",   # Winter: 温跃层较深或减弱
    1: "45–105 m",   # Spring: 上移
    2: "25–85 m",    # Summer: 最浅
    3: "45–125 m",   # Autumn: 下移
}

# ── CTD 数据集构建参数 ────────────────────────────────────────────────────────
CTD_TSF_AUV_NOISE_STD    = 7.0    # AUV 位置相对温跃层的高斯偏差（米）
CTD_TSF_NEAR_THERMO_PROB = 0.5    # AUV 采样在温跃层附近的概率
CTD_TSF_STRIDE           = 5      # 滑窗步长
CTD_TSF_N_AUGMENT        = 12     # 每个滑窗位置重复采样次数（不同 AUV 噪声）
MAX_SAMPLES_PER_EPOCH    = 16800   # 15%×48000；few-shot 实验；100%=48000, 75%=36000, 35%=16800, 15%=7200
CTD_AUV_EXCLUDE_KEYWORDS = ('AUV_',)  # 排除 AUV 采集文件
CTD_YEAR_BLOCK_SIZE      = 3      # 兼容旧版数据划分参数（主流程已不使用）
CTD_YEAR_WM_COUNT        = 2

# ── CTD 数据集划分清单路径（由 generate_split_manifest.py 生成）────────────────
CTD_SPLIT_MANIFEST = os.getenv(
    "THERMOCAST_SPLIT_MANIFEST",
    os.path.join(CTD_FOLDER_PATH, "split_manifest.json"),
)

# ── 数据集路径 ────────────────────────────────────────────────────────────────
# 由 RL 仿真（DQNSeqPomdp_VLMdataGen）生成的旧版 TSF 数据集
TSF_DATA_PATH = os.getenv("THERMOCAST_TSF_DATA_PATH", "data/ThermoTSF_dataset")

# CTDTSFDataset 磁盘缓存目录（None 表示禁用）
# 首次构建后序列化为 .pkl，后续运行直接加载，跳过 .mat 读取和插值计算
# 键含所有影响样本内容的参数及文件 mtime，数据或参数变化时自动失效
DATASET_CACHE_DIR = "dataset_cache"

# ── 训练通用超参数 ────────────────────────────────────────────────────────────
EPOCHS           = 100   # 最大训练轮数；early stopping 可提前终止
GRAD_ACCUM_STEPS = 4     # 梯度累积步数；等效 batch = BATCH_SIZE × GRAD_ACCUM_STEPS
ES_PATIENCE      = 10    # early stopping：连续多少 epoch val_mae 无改善则终止
ES_MIN_EPOCHS    = 50    # early stopping 最早触发的 epoch；未达到此轮数则延迟中断
CKPT_INTERVAL    = 10    # 定期检查点间隔（epoch）；每次覆盖上一个，磁盘只保留一份
WEIGHT_DECAY     = 1e-3  # AdamW weight decay（LoRA 参数组固定为 0）
HUBER_DELTA      = 0.2   # Huber loss 的 delta（归一化空间，约对应 12.5m 实际深度）

# ── GPU ────────────────────────────────────────────────────────────────────────
CUDA_DEVICE_LLM = "1"
CUDA_DEVICE_DQN = "0"

# ── 日志 ──────────────────────────────────────────────────────────────────────
SWANLAB_PROJECT    = "ThermoTSF-Reprogram"
SWANLAB_LOG_INTERVAL = 50   # 每隔多少 step 打印并上传一次窗口 loss
