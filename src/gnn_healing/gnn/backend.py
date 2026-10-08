"""Hardware / library backend detection.

Full-batch PyG on a single device is the reference path (the twin here is small).
When cugraph-pyg and WholeGraph are installed (docs/nvidia.md) the same model classes
run on GPU-resident graph structure with NVIDIA's samplers; see gnn/scale.py.
"""
from __future__ import annotations
import importlib
import torch


def describe_backend() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    for mod in ("torch_geometric", "cugraph_pyg", "pylibwholegraph", "cudf"):
        try:
            m = importlib.import_module(mod)
            info[mod] = getattr(m, "__version__", "present")
        except Exception:
            info[mod] = None
    return info


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
