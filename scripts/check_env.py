from __future__ import annotations

import os
import platform
from pathlib import Path

import torch


def main() -> None:
    print("python:", platform.python_version())
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_runtime:", torch.version.cuda)
        print("gpu:", torch.cuda.get_device_name(0))
    for name in [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "REWARD_MODELS_DIR",
        "REWARD_DATASETS_DIR",
        "REWARD_OUTPUTS_DIR",
    ]:
        value = os.environ.get(name, "")
        print(f"{name}: {value}")
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()

