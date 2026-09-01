# ThermoTSF-Reprogram

This repository contains **ThermoTSF-Reprogram**, the large-language-model-based time-series forecaster used by the **Forecast-to-Reward (F2R)** framework for adaptive thermocline sampling with autonomous underwater vehicles (AUVs).

ThermoTSF-Reprogram predicts short-horizon thermocline-center depths from partial, trajectory-dependent conductivity-temperature-depth (CTD) observations. It maps continuous depth, temperature, salinity, and contextual variables into the representation space of a pretrained Qwen3.5-2B model without converting the numerical profiles into text. The forecasts shape rewards during offline F2R policy learning; the forecaster does not select AUV actions and is not deployed onboard.

The companion reinforcement-learning and deployment implementation is available in [F2R policy learning](https://github.com/tianzhuoer/L2R).

## Forecasting task

At each prediction time, the model receives the preceding `K = 32` AUV sampling steps and predicts the thermocline-center depth for the following `H = 5` steps:

```text
[depth, partial temperature profile, partial salinity profile] x 32
                    + seasonal and hydrographic context
                                      |
                                      v
                  five future thermocline-center depths
```

Each partial temperature and salinity profile contains 15 depth bins accumulated from measurements along the simulated AUV trajectory. The complete CTD profile is used only to derive future-depth labels.

The reference thermocline-center depth is the midpoint of the adjacent depth interval with the largest absolute vertical temperature gradient within the physically relevant depth range of 25--150 m. Output uses the AUV convention of negative depth below the sea surface.

## Architecture

ThermoTSF-Reprogram preserves both modality structure and the full temporal resolution of the observation history:

1. **Modality-specific encoding.** AUV depth, partial temperature, partial salinity, and a nine-dimensional context vector are projected separately.
2. **Cross-attention reprogramming.** Trainable modality-specific layers align numerical representations with prototypes in the Qwen embedding space.
3. **Temporally interleaved numerical tokens.** Depth, temperature, and salinity tokens are interleaved by sampling step rather than collapsed across time.
4. **Seasonal and hydrographic context.** A season-aware task prompt is combined with a reprogrammed context token containing season, day-of-year encodings, and summary statistics of the CTD window.
5. **LoRA-adapted backbone.** Prompt and numerical tokens are processed jointly by Qwen3.5-2B with low-rank adaptation (LoRA).
6. **Multi-step regression.** Last and mean pooled numerical-token hidden states are passed to an MLP head that predicts five future depths.

```text
depth history --------> projection --+
temperature profiles -> projection --+--> prototype cross-attention
salinity profiles ----> projection --+             |
context --------------> projection --+             v
                                             reprogrammed tokens
season-aware prompt ---------------------------------+
                                                      v
                                             LoRA-adapted Qwen
                                                      v
                                       numeric-token last/mean pooling
                                                      v
                                         five-step regression head
```

The model is optimized in normalized depth space using Huber loss. Gaussian perturbations are applied to depth, temperature, and salinity during training to improve robustness to sensor noise and interpolation errors.

## Repository structure

```text
.
|-- train_reprogram_TSF.py       # ThermoTSF-Reprogram entry point
|-- train_timellm.py             # Time-LLM comparison
|-- train_chronos2.py            # Chronos-2 comparison
|-- train_timesfm.py             # TimesFM comparison
|-- train_itransformer.py        # iTransformer comparison
|-- train_transformer.py         # Transformer comparison
|-- train_lstm.py                # LSTM comparison
|-- train_llm_TSF.py             # text-based LLM comparison
|-- configs/                     # shared and model-specific settings
|-- models/                      # forecasters, baselines, and wrappers
|-- trainers/                    # common training and validation loop
`-- tsf_data/                    # datasets, collation, and metrics
```

Data, pretrained weights, checkpoints, logs, and figures are excluded from version control.

## Installation

```bash
pip install -r requirements.txt
```

The main model requires Qwen3.5-2B, either from a local directory or through the configured Hugging Face identifier.

## Data preparation

```bash
export THERMOCAST_CTD_PATH=/path/to/CTD
export THERMOCAST_QWEN_PATH=/path/to/Qwen3.5-2B
export THERMOCAST_SPLIT_MANIFEST=/path/to/CTD/split_manifest.json
```

PowerShell:

```powershell
$env:THERMOCAST_CTD_PATH = "D:\datasets\CTD"
$env:THERMOCAST_QWEN_PATH = "D:\models\Qwen3.5-2B"
$env:THERMOCAST_SPLIT_MANIFEST = "D:\datasets\CTD\split_manifest.json"
```

The forecasting and policy-learning repositories share a file-level split manifest. Complete CTD files used for forecaster fine-tuning must remain disjoint from files used to construct F2R policy-training environments. A season-balanced manifest can be generated with `generate_split_manifest.py` in the [F2R repository](https://github.com/tianzhuoer/L2R).

The forecasting dataset excludes AUV field records by default (`AUV_` filename prefix). In the manuscript, Argo and NOAA CTD records were used for forecaster development, while AUV field observations were reserved for experimental comparison.

## Training

Single GPU:

```bash
python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 0
```

Multiple GPUs:

```bash
accelerate launch --num_processes=<N> train_reprogram_TSF.py --mode lora --samples 16800
```

| Setting | Default |
|---|---:|
| Input history `K` | 32 steps |
| Forecast horizon `H` | 5 steps |
| Partial profile size | 15 bins per modality |
| Target depth range | `[-150, -25] m` |
| Reprogramming dimension | 64 |
| Cross-attention heads | 4 |
| Prototypes | 128 |
| LoRA rank / alpha | 8 / 16 |
| Batch size / gradient accumulation | 8 / 8 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 epochs, after epoch 50 |

Best checkpoints are written under `checkpoints/reprogram/<variant>-<samples>k/` as `lora_best/` and `head_best.pt`.

## Ablations and comparison models

```bash
# Frozen Qwen backbone; train reprogramming layers and head
python train_reprogram_TSF.py --mode head --samples 16800 --gpu 0

# Replace prototype cross-attention with linear projection
python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 0 --no-reprogram

# Skip the language-model backbone
python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 0 --no-backbone

# Randomly initialize the Qwen architecture
python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 0 --random-init

# Add window-specific statistics to the textual prompt
python train_reprogram_TSF.py --mode lora --samples 16800 --gpu 0 --dynamic-prompt
```

Comparison entry points include:

```bash
python train_timellm.py --gpu 0
python train_chronos2.py --gpu 0
python train_lstm.py --gpu 0
python train_itransformer.py --gpu 0
```

## Relationship to F2R

ThermoTSF-Reprogram is trained before F2R policy adaptation and then frozen. During policy learning, its five-step predictions are screened, smoothed, and aggregated to construct a forecast depth for shaped rewards. It does not choose actions. After F2R training, the forecaster and the additional KL reference-network copy are removed; only the lightweight attention-based policy and residual Q-adapter are deployed.

## Scope and reproducibility

- The task predicts a single maximum-gradient thermocline-center depth; it does not estimate thickness, intensity, or multilayer structure.
- Manuscript data are concentrated in the South China Sea and do not represent global thermocline variability.
- Reported forecasting results are validation results from the manuscript's current setting.
- Data and model weights are not included. Users must provide CTD files and verify all applicable licenses.
- This repository currently has no explicit software license. Obtain permission before reuse.

## Citation

To be updated.
