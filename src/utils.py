import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class NaNLossError(Exception):
    """Raised when a loss becomes NaN during training."""


class TrialExecutionError(Exception):
    """Raised when a trial fails in a recoverable way."""


def _configure_cpu_torch_threads() -> int:
    """
    Some CPU environments can hang badly on this project with PyTorch's default thread count
    during transformer encoding on sparse WiFi tensors. We prefer a conservative default of 1
    for correctness/stability; callers can override it with WIFI_TORCH_CPU_THREADS.
    """
    raw = os.environ.get("WIFI_TORCH_CPU_THREADS", "1")
    try:
        threads = max(1, int(raw))
    except ValueError:
        threads = 1
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(threads, 1)))
    except Exception:
        pass
    return threads


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
    strict_deterministic_algorithms: bool = False,
    deterministic_warn_only: bool = True,
    cublas_workspace_config: str = ":4096:8",
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", cublas_workspace_config)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if strict_deterministic_algorithms:
            try:
                torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
            except TypeError:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        else:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def get_device(require_cuda: bool = True, device_index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_index}")
        print(f"[设备] 当前使用: {device}")
        print(f"[设备] GPU 名称: {torch.cuda.get_device_name(device_index)}")
        return device

    if require_cuda:
        raise RuntimeError("配置要求使用 CUDA，但当前 PyTorch 未检测到可用 CUDA。")

    device = torch.device("cpu")
    cpu_threads = _configure_cpu_torch_threads()
    print(f"[设备] 当前使用: {device}")
    print(f"[设备] CPU torch threads={cpu_threads}")
    return device


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}小时{m}分{s}秒"
    if m > 0:
        return f"{m}分{s}秒"
    return f"{s}秒"


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: os.PathLike[str] | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(data: Dict[str, Any], path: os.PathLike[str] | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def copy_dir(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def wall_clock() -> float:
    return time.time()
