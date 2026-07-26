import argparse
import os


def apply_gpu_arg(default: str = "0") -> str:
    """
    从命令行解析 --gpu 参数并设置 CUDA_VISIBLE_DEVICES。

    用法示例:
        python train_reprogram_TSF.py --gpu 1
        python train_reprogram_TSF.py --gpu 0,1   # 多卡
        accelerate launch --num_processes=2 train_reprogram_TSF.py --gpu 0,1

    若不传 --gpu，则使用 default（来自 cfg.CUDA_DEVICE_LLM）。
    必须在 Accelerator() 初始化之前调用。
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", default=None, help="GPU id(s), e.g. 0 or 1 or 0,1")
    args, _ = p.parse_known_args()
    gpu = args.gpu if args.gpu is not None else default
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu


def apply_samples_arg(cfg) -> None:
    """
    从命令行解析 --samples 参数，覆盖 cfg.MAX_SAMPLES_PER_EPOCH。
    同时更新 cfg.SWANLAB_EXPERIMENT 使其包含数据量标识。

    用法示例:
        python train_lstm.py --samples 7200
        python train_lstm.py --samples 36000
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--samples", type=int, default=None,
                   help="覆盖 MAX_SAMPLES_PER_EPOCH，如 --samples 7200")
    args, _ = p.parse_known_args()
    if args.samples is not None:
        cfg.MAX_SAMPLES_PER_EPOCH = args.samples
        # 重新生成 experiment name 使其反映实际数据量
        base = cfg.SWANLAB_EXPERIMENT
        # 去掉已有的 -<数字>k 后缀再重新拼
        import re
        base = re.sub(r'-\d+k$', '', base)
        cfg.SWANLAB_EXPERIMENT = f"{base}-{cfg.MAX_SAMPLES_PER_EPOCH//1000}k"
