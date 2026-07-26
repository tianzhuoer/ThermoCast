# ThermoCast

ThermoCast 使用 AUV 历史深度、温度剖面、盐度剖面和季节信息，预测未来多个时刻的温跃层中心深度。核心方法将连续海洋观测重编程到预训练语言模型的嵌入空间，并使用 LoRA 完成参数高效微调。

## 项目结构

```text
ThermoCast/
├── configs/                 # 数据、训练及各模型配置
├── models/                  # Reprogram-TSF、Time-LLM 与基线网络
├── trainers/                # 通用训练与验证流程
├── tsf_data/                # CTD 数据集、特征整理和指标
├── train_reprogram_TSF.py   # 主方法训练入口
├── train_*.py               # 其他模型与基线训练入口
└── requirements.txt
```

数据、预训练权重、检查点、日志和图片不包含在本仓库中。

## 网络框架

主方法 Reprogram-TSF 在每个时间步分别编码深度、15 维温度剖面和 15 维盐度剖面，保留完整时间分辨率。各模态经过独立线性投影和 prototype cross-attention，被映射到 LLM embedding 空间，并按时间交错组成数值 token。

季节 prompt 提供领域语义，归一化季节与年内日期通过 `[THERMO]` 位置注入。拼接后的 prompt token 与数值 token 输入 Qwen backbone；backbone 使用 LoRA 微调，最后对数值 token 执行 last/mean pooling，通过 MLP 回归未来 `H` 步温跃层深度。

仓库还提供 Time-LLM、文本式 LLM-TSF、Chronos-2、TimesFM、iTransformer、Transformer 和 LSTM 对照模型。

## 使用

安装依赖：

```bash
pip install -r requirements.txt
```

设置 CTD 数据目录和本地 Qwen 模型目录：

```bash
# Linux/macOS
export THERMOCAST_CTD_PATH=/path/to/CTD
export THERMOCAST_QWEN_PATH=/path/to/Qwen3.5-2B

# PowerShell
$env:THERMOCAST_CTD_PATH="D:\path\to\CTD"
$env:THERMOCAST_QWEN_PATH="D:\path\to\Qwen3.5-2B"
```

训练主方法：

```bash
python train_reprogram_TSF.py --gpu 0
```

训练基线：

```bash
python train_lstm.py --gpu 0
python train_itransformer.py --gpu 0
python train_chronos2.py --gpu 0
```

数据窗口、预测长度、训练样本数和其他超参数可在 `configs/` 中修改。训练生成的检查点默认写入 `checkpoints/`。
