from .shared import *

# ── Qwen 模型路径 ─────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("THERMOCAST_QWEN_PATH", "Qwen/Qwen3.5-2B")
# ── 训练超参数 ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 16    # 每个 GPU 的 batch size（LLM 显存占用大，比 TimesFM 小）
MAX_LENGTH = 380   # tokenizer 截断长度；compact 格式约 270–300 token，留余量

# ── 输入格式消融 ───────────────────────────────────────────────────────────────
# "full": 20 步观测全部展开为紧凑数值表
# "compact": 早期历史用统计摘要，最近若干步用定点整数编码
TSF_INPUT_FORMAT       = "compact"  # "full" | "compact"，控制观测序列编码方式
COMPACT_RECENT_STEPS   = 10         # compact 模式下保留完整逐步观测的最近步数
COMPACT_KEEP_SALINITY  = True       # compact 模式是否保留盐度特征列

# ── LoRA 配置 ─────────────────────────────────────────────────────────────────
LORA_R       = 8    # LoRA 秩；越大可学习参数越多，显存消耗越高
LORA_ALPHA   = 16   # 缩放系数，通常设为 2 × LORA_R
LORA_DROPOUT = 0.05 # LoRA 层 dropout，防过拟合
LORA_TARGET  = ["k_proj", "q_proj", "v_proj", "o_proj"]  # 注入 LoRA 的注意力投影层

# ── 学习率 ────────────────────────────────────────────────────────────────────
LR_LORA = 5e-6
LR_HEAD = 5e-6   # 与 LR_LORA 等齐，防止 head 先收敛锁死 LoRA

# ── 系统提示（{season}/{H} 在 collate_fn 中替换）────────────────────────────
PROMPT_TSF = (
    "You are an expert oceanographer. "
    "Predict thermocline center depth for the next {H} timesteps "
    "from AUV observations in {season}."
)

# ── 训练过程噪声增强 ──────────────────────────────────────────────────────────
DEPTH_NOISE_STD = 1.0    # 输入深度噪声（米）
FEAT_NOISE_STD  = 0.01   # 温盐观测噪声

# ── 检查点 ─────────────────────────────────────────────────────────────────────
TSF_CKPT_DIR  = f"checkpoints/llm_{TSF_INPUT_FORMAT}"
TSF_LORA_BEST = f"{TSF_CKPT_DIR}/lora_best"
TSF_HEAD_BEST = f"{TSF_CKPT_DIR}/head_best.pt"

# ── SwanLab ────────────────────────────────────────────────────────────────────
SWANLAB_EXPERIMENT = f"qwen3.5-0.8b-lora-tsf-{TSF_INPUT_FORMAT}"
