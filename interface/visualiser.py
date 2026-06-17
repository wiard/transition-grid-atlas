"""Pure IO rendering for Experimental Quantum & RTT Lab artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np

from engine.trajectories import load_trajectory_npz


def load_probability_artifact(artifact_path: Path) -> dict[str, Any]:
    """Load a saved trajectory artifact from ``results/trajectories``."""
    artifact = load_trajectory_npz(artifact_path)
    metadata = dict(artifact.get("metadata", {}))
    return {
        "run_id": str(metadata.get("run_id", Path(artifact_path).stem.removeprefix("run_"))),
        "theory_mode": str(metadata.get("theory_mode", metadata.get("mode", "unknown"))),
        "probability_frames": np.asarray(artifact["probabilities"], dtype=np.float64),
        "times": np.asarray(artifact["times"], dtype=np.float64),
        "metadata": metadata,
        "config_hash": str(artifact["config_hash"]),
    }


def animate_probability_trajectory(
    probabilities: np.ndarray,
    output_path: str | Path,
    *,
    interval_ms: int = 40,
    title: str = "Kinetic Transition Atlas",
) -> Path:
    """Render p(x,t) to GIF or MP4 from an already prepared trajectory."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_times, L).")

    n_times, L = probabilities.shape
    x = np.arange(L, dtype=np.float64)

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, probabilities[0])
    ax.set_xlim(0, L - 1)
    ax.set_ylim(0.0, max(float(np.max(probabilities)) * 1.1, 1.0e-12))
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\Psi(x,t)|^2$ or $\rho_{xx}(t)$")
    ax.set_title(title)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame: int) -> tuple[Any, ...]:
        line.set_ydata(probabilities[frame])
        time_text.set_text(f"t = {frame}")
        return line, time_text

    anim = FuncAnimation(fig, update, frames=n_times, interval=interval_ms, blit=True)

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        anim.save(output_path, writer="pillow")
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required to save MP4 animations")
        anim.save(output_path)
    else:
        raise ValueError("output_path must end in .gif or .mp4")

    plt.close(fig)
    return output_path


def render_probability_animation(
    *,
    artifact_path: Path,
    output_path: Path,
    fps: int = 30,
) -> Path:
    """Backward-compatible artifact consumer for the CLI."""

    artifact = load_probability_artifact(artifact_path)
    return animate_probability_trajectory(
        artifact["probability_frames"],
        output_path,
        interval_ms=max(1, int(round(1000.0 / max(1, fps)))),
        title=f"{artifact['theory_mode']} · run {artifact['run_id']}",
    )
