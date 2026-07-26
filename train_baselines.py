"""
顺序训练三个无预训练基线：LSTM → Transformer → iTransformer。

用法
----
单 GPU:  python train_baselines.py
多 GPU:  accelerate launch --num_processes=<N> train_baselines.py

某个模型训练失败时，打印错误并继续下一个（不中断整体流程）。
"""

import subprocess
import sys
import time

BASELINES = [
    ("LSTM",          "train_lstm.py"),
    ("Transformer",   "train_transformer.py"),
    ("iTransformer",  "train_itransformer.py"),
]


def run(name: str, script: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  开始训练：{name}  ({script})")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "完成" if ok else f"失败（exit {result.returncode}）"
    print(f"\n[{name}] {status}  耗时 {elapsed/60:.1f} min")
    return ok


if __name__ == "__main__":
    results = {}
    for name, script in BASELINES:
        results[name] = run(name, script)

    print(f"\n{'='*60}")
    print("  训练汇总")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    print()
