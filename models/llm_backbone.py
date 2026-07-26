import os
import pathlib
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _local_path(p: str) -> str:
    """把 Windows 反斜杠路径转为正斜杠，避免新版 huggingface_hub 把本地路径当 repo id 校验失败。"""
    return pathlib.Path(p).as_posix()


def load_backbone(cfg, random_init: bool = False):
    """加载 Qwen 基座模型并注入 LoRA adapter，返回 (peft_model, tokenizer)。

    random_init=True 时：用同一 config 随机初始化权重（结构完全一致，但不加载
    预训练参数），用于"预训练知识是否起作用"的诊断消融。tokenizer 仍用真的，
    保证只改变"权重是否预训练"这一个变量。
    """
    model_path = _local_path(cfg.MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if random_init:
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        base_model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        print("[random_init] backbone 使用随机初始化权重（结构同 Qwen，无预训练知识）")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map=None,
            dtype=torch.bfloat16,
            torch_dtype="auto",
        )
    print(f"base_model device: {next(base_model.parameters()).device}")
    base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, tokenizer


def _pool(hidden_states, attention_mask):
    """[last_token ; mean_pool] 拼接，返回 (B, 2*hidden_size)。"""
    if attention_mask is not None:
        mask      = attention_mask.unsqueeze(-1).float()
        mean_pool = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        seq_len   = attention_mask.sum(dim=1) - 1
        last_pool = hidden_states[torch.arange(hidden_states.size(0),
                                               device=hidden_states.device), seq_len]
    else:
        mean_pool = hidden_states.mean(dim=1)
        last_pool = hidden_states[:, -1, :]
    return torch.cat([last_pool, mean_pool], dim=-1)


class ThermoForecaster(nn.Module):
    """
    温跃层中心深度时序预测头（回归）。

    backbone 通过 LoRA 微调；回归头全量训练。
    输出 H 个未来时间步的归一化深度（[-1, 1]）。
    """

    def __init__(self, peft_model, H: int):
        super().__init__()
        self.backbone = peft_model.model
        self.H = H
        hidden_size = peft_model.config.hidden_size
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, H),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        pooled = _pool(outputs.hidden_states[-1], attention_mask)
        return self.regressor(pooled.to(self.regressor[0].weight.dtype))  # (B, H)
