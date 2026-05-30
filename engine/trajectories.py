"""Trajectory artifact IO for the Kinetic Transition Atlas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_trajectory_npz(
    path: str | Path,
    *,
    probabilities: np.ndarray,
    times: np.ndarray,
    x_mean: np.ndarray,
    x_var: np.ndarray,
    metadata: dict[str, Any],
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    digest = config_hash(metadata)
    payload: dict[str, Any] = {
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "times": np.asarray(times, dtype=np.float64),
        "x_mean": np.asarray(x_mean, dtype=np.float64),
        "x_var": np.asarray(x_var, dtype=np.float64),
        "metadata": json.dumps(metadata, sort_keys=True),
        "config_hash": digest,
    }
    if extra_arrays:
        for key, value in extra_arrays.items():
            payload[key] = np.asarray(value)

    np.savez_compressed(path, **payload)
    return digest


def load_trajectory_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "probabilities": np.asarray(data["probabilities"], dtype=np.float64),
            "times": np.asarray(data["times"], dtype=np.float64),
            "x_mean": np.asarray(data["x_mean"], dtype=np.float64),
            "x_var": np.asarray(data["x_var"], dtype=np.float64),
            "metadata": json.loads(str(data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"])),
            "config_hash": str(data["config_hash"].item() if hasattr(data["config_hash"], "item") else data["config_hash"]),
            **{
                key: np.asarray(data[key])
                for key in data.files
                if key not in {"probabilities", "times", "x_mean", "x_var", "metadata", "config_hash"}
            },
        }
