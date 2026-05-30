from __future__ import annotations

import argparse
import copy
import csv
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.lab_modes import build_lab_config, run_lab_simulation
from run import OUTPUTS_DIR, LAB_TRAJECTORIES_DIR, ensure_output_dirs, load_config
from validation.kta_audit import summarize_kta_audit

BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "lab" / "qm_free_demo.yaml"
GAMMA_VALUES = [0.00, 0.01, 0.02, 0.05, 0.10, 0.50, 1.00]
FIXED_W = 1.5
FIXED_MODE = "lindblad"
PLOT_OUTPUT_PATH = PROJECT_ROOT / "results" / "renders" / "lindblad_enaqt_profile.png"
GRID_W_VALUES = [0.0, 0.5, 1.5, 3.0]
GRID_GAMMA_VALUES = [0.00, 0.02, 0.05, 0.50, 1.00]
GRID_CSV_PATH = OUTPUTS_DIR / "lindblad_2d_grid_sweep.csv"
GRID_PLOT_OUTPUT_PATH = PROJECT_ROOT / "results" / "renders" / "lindblad_phase_atlas.png"


def timestamp_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def gamma_token(gamma: float) -> str:
    return f"{gamma:.2f}".replace(".", "p")


def disorder_token(disorder_strength: float) -> str:
    return f"{disorder_strength:.2f}".replace(".", "p")


def build_sweep_config(
    base_config: dict[str, Any],
    gamma: float,
    disorder_strength: float,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    system_cfg = dict(config.get("system", {}))
    system_cfg["mode"] = FIXED_MODE
    system_cfg["disorder_strength"] = float(disorder_strength)
    system_cfg["gamma"] = float(gamma)
    config["system"] = system_cfg
    return config


def collect_row(base_config: dict[str, Any], gamma: float, disorder_strength: float) -> dict[str, Any]:
    config = build_sweep_config(base_config, gamma, disorder_strength)
    lab_cfg = build_lab_config(config, mode_override=FIXED_MODE, gamma_override=gamma)
    run_id = (
        f"{timestamp_token()}_{FIXED_MODE}_"
        f"W_{disorder_token(disorder_strength)}_gamma_{gamma_token(gamma)}"
    )

    result = run_lab_simulation(
        config=config,
        run_id=run_id,
        trajectories_dir=LAB_TRAJECTORIES_DIR,
        lab_config=lab_cfg,
    )
    audit = summarize_kta_audit(
        probability_frames=result["probability_frames"],
        times=result["times"],
        trace_series=result["trace_series"],
        x0=lab_cfg.x0,
        config=lab_cfg,
        coherence_norm_final=result["summary_row"]["coherence_norm_final"],
    )

    return {
        "run_id": run_id,
        "mode": FIXED_MODE,
        "W": float(disorder_strength),
        "gamma": float(gamma),
        "trace_error": float(audit["trace_error"]),
        "alpha_late": float(audit["alpha_late"]),
        "r2_late": float(audit["r2_late"]),
        "ipr_final": float(audit["ipr_final"]),
        "edge_hit": bool(audit["edge_hit"]),
        "falldown_candidate": bool(audit["falldown_candidate"]),
        "falldown_score": float(audit["falldown_score"]),
        "frame_corr_late": float(audit["frame_corr_late"]),
        "x_var_drift_late": float(audit["x_var_drift_late"]),
        "coherence_norm_final": float(audit["coherence_norm_final"]),
        "zeno_indicator": bool(audit["zeno_indicator"]),
        "trajectory_path": str(result["artifact_path"]),
        "artifact_hash": str(result["artifact_hash"]),
        "config_hash": str(result["config_hash"]),
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}.")
    return rows


def latest_sweep_csv() -> Path:
    csv_paths = sorted(OUTPUTS_DIR.glob("lindblad_gamma_sweep_*.csv"))
    if not csv_paths:
        raise FileNotFoundError("No Lindblad sweep CSV found in outputs/.")
    return csv_paths[-1]


def build_grid_rows(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for disorder_strength in GRID_W_VALUES:
        for gamma in GRID_GAMMA_VALUES:
            rows.append(collect_row(base_config, gamma, disorder_strength))
    return rows


def plot_sweep(csv_path: Path, output_path: Path) -> Path:
    rows = read_csv_rows(csv_path)
    gammas = [float(row["gamma"]) for row in rows]
    alpha_vals = [float(row["alpha_late"]) for row in rows]
    ipr_vals = [float(row["ipr_final"]) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_alpha, ax_ipr) = plt.subplots(
        2,
        1,
        figsize=(9, 7.5),
        sharex=True,
        constrained_layout=True,
    )

    for axis in (ax_alpha, ax_ipr):
        axis.set_xscale("symlog", linthresh=0.01, linscale=1.1)
        axis.axvspan(-0.0005, 0.01, color="#6C7A89", alpha=0.12)
        axis.axvspan(0.01, 0.10, color="#7CB342", alpha=0.12)
        axis.axvspan(0.10, 1.00, color="#F9A825", alpha=0.12)

    ax_alpha.plot(gammas, alpha_vals, marker="o", markersize=6, linewidth=2.2, color="#0B5FFF")
    ax_alpha.axhline(0.5, linestyle="--", linewidth=1.5, color="#555555")
    ax_alpha.set_ylabel(r"$\alpha_{late}$")
    ax_alpha.set_title("Lindblad ENAQT Profile")
    ax_alpha.grid(True, alpha=0.3, which="both")
    ax_alpha.text(
        0.003,
        0.94,
        "Localized",
        transform=ax_alpha.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        color="#39424E",
        fontweight="semibold",
    )
    ax_alpha.text(
        0.04,
        0.94,
        "Assisted Transport",
        transform=ax_alpha.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        color="#2F5D2F",
        fontweight="semibold",
    )
    ax_alpha.text(
        0.42,
        0.94,
        "Near-Diffusive Window",
        transform=ax_alpha.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        color="#8A5A00",
        fontweight="semibold",
    )

    ax_ipr.plot(gammas, ipr_vals, marker="o", markersize=6, linewidth=2.2, color="#C04B00")
    ax_ipr.set_xlabel(r"$\gamma$")
    ax_ipr.set_ylabel(r"$IPR_{final}$")
    ax_ipr.grid(True, alpha=0.3, which="both")

    tick_values = [0.00, 0.01, 0.02, 0.05, 0.10, 0.50, 1.00]
    ax_ipr.set_xlim(-0.001, 1.1)
    ax_ipr.set_xticks(tick_values)
    ax_ipr.set_xticklabels([f"{value:.2f}" for value in tick_values])
    ax_ipr.tick_params(axis="x", labelsize=9)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_grid_heatmap(csv_path: Path, output_path: Path) -> Path:
    rows = read_csv_rows(csv_path)
    alpha_map = {
        (float(row["W"]), float(row["gamma"])): float(row["alpha_late"])
        for row in rows
    }
    ipr_map = {
        (float(row["W"]), float(row["gamma"])): float(row["ipr_final"])
        for row in rows
    }

    alpha_matrix = []
    ipr_matrix = []
    for disorder_strength in GRID_W_VALUES:
        alpha_matrix.append(
            [alpha_map[(disorder_strength, gamma)] for gamma in GRID_GAMMA_VALUES]
        )
        ipr_matrix.append(
            [ipr_map[(disorder_strength, gamma)] for gamma in GRID_GAMMA_VALUES]
        )

    alpha_matrix_np = np.asarray(alpha_matrix, dtype=float)
    ipr_matrix_np = np.asarray(ipr_matrix, dtype=float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_alpha, ax_ipr) = plt.subplots(
        1,
        2,
        figsize=(13, 4.8),
        constrained_layout=True,
    )

    alpha_image = ax_alpha.imshow(
        alpha_matrix_np,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=float(np.min(alpha_matrix_np)),
        vmax=float(np.max(alpha_matrix_np)),
    )

    x_positions = np.arange(len(GRID_GAMMA_VALUES), dtype=float)
    y_positions = np.arange(len(GRID_W_VALUES), dtype=float)
    grid_x, grid_y = np.meshgrid(x_positions, y_positions)

    for y_index, disorder_strength in enumerate(GRID_W_VALUES):
        for x_index, gamma in enumerate(GRID_GAMMA_VALUES):
            alpha_val = alpha_map[(disorder_strength, gamma)]
            text_color = "white" if alpha_val < 0.45 else "black"
            ax_alpha.text(
                x_index,
                y_index,
                f"{alpha_val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="semibold",
            )

    contour = ax_alpha.contour(
        grid_x,
        grid_y,
        alpha_matrix_np,
        levels=[0.50],
        colors=["white"],
        linewidths=2.0,
    )
    ax_alpha.clabel(contour, fmt={0.50: r"$\alpha=0.50$"}, inline=True, fontsize=8)

    ax_alpha.set_xticks(x_positions)
    ax_alpha.set_xticklabels([f"{value:.2f}" for value in GRID_GAMMA_VALUES])
    ax_alpha.set_yticks(y_positions)
    ax_alpha.set_yticklabels([f"{value:.1f}" for value in GRID_W_VALUES])
    ax_alpha.set_xlabel(r"Dephasing $\gamma$")
    ax_alpha.set_ylabel(r"Disorder $W$")
    ax_alpha.set_title(r"$\alpha_{late}$ Phase Map")

    alpha_colorbar = fig.colorbar(alpha_image, ax=ax_alpha, pad=0.02)
    alpha_colorbar.set_label(r"$\alpha_{late}$")
    alpha_colorbar.ax.axhline(0.5, color="white", linestyle="--", linewidth=1.5)

    ipr_image = ax_ipr.imshow(
        ipr_matrix_np,
        origin="lower",
        aspect="auto",
        cmap="plasma",
        vmin=float(np.min(ipr_matrix_np)),
        vmax=float(np.max(ipr_matrix_np)),
    )

    for y_index, disorder_strength in enumerate(GRID_W_VALUES):
        for x_index, gamma in enumerate(GRID_GAMMA_VALUES):
            ipr_val = ipr_map[(disorder_strength, gamma)]
            text_color = "white" if ipr_val > 0.04 else "black"
            ax_ipr.text(
                x_index,
                y_index,
                f"{ipr_val:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="semibold",
            )

    ax_ipr.set_xticks(x_positions)
    ax_ipr.set_xticklabels([f"{value:.2f}" for value in GRID_GAMMA_VALUES])
    ax_ipr.set_yticks(y_positions)
    ax_ipr.set_yticklabels([f"{value:.1f}" for value in GRID_W_VALUES])
    ax_ipr.set_xlabel(r"Dephasing $\gamma$")
    ax_ipr.set_ylabel(r"Disorder $W$")
    ax_ipr.set_title(r"$IPR_{final}$ Localization Map")

    ipr_colorbar = fig.colorbar(ipr_image, ax=ax_ipr, pad=0.02)
    ipr_colorbar.set_label(r"$IPR_{final}$")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def print_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    print("Lindblad gamma sweep")
    print(f"base_config: {BASE_CONFIG_PATH}")
    print(f"mode: {FIXED_MODE}")
    print(f"W: {FIXED_W:.2f}")
    print(f"output_csv: {output_path}")
    print("")
    print("gamma    alpha_late   r2_late   ipr_final   edge_hit   zeno")
    for row in rows:
        print(
            f"{row['gamma']:>5.2f}    "
            f"{row['alpha_late']:>10.6f}   "
            f"{row['r2_late']:>7.6f}   "
            f"{row['ipr_final']:>9.6f}   "
            f"{str(row['edge_hit']):>8}   "
            f"{str(row['zeno_indicator']):>5}"
        )


def print_grid_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    print("Lindblad 2D grid sweep")
    print(f"base_config: {BASE_CONFIG_PATH}")
    print(f"mode: {FIXED_MODE}")
    print(f"output_csv: {output_path}")
    print("")
    print("W      gamma    alpha_late   ipr_final   edge_hit   zeno")
    for row in rows:
        print(
            f"{row['W']:>3.1f}    "
            f"{row['gamma']:>5.2f}    "
            f"{row['alpha_late']:>10.6f}   "
            f"{row['ipr_final']:>9.6f}   "
            f"{str(row['edge_hit']):>8}   "
            f"{str(row['zeno_indicator']):>5}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and plot Lindblad gamma sweeps.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the sweep run and plot the newest matching CSV in outputs/.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Plot a specific Lindblad sweep CSV instead of the newest one.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=PLOT_OUTPUT_PATH,
        help="Where to write the PNG profile plot.",
    )
    parser.add_argument(
        "--grid2d",
        action="store_true",
        help="Run the 2D W-gamma sweep and render the heatmap atlas.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    csv_path: Path

    if args.grid2d:
        base_config = load_config(BASE_CONFIG_PATH)
        rows = build_grid_rows(base_config)
        csv_path = GRID_CSV_PATH
        write_csv(rows, csv_path)
        print_grid_summary(rows, csv_path)
        plot_path = plot_grid_heatmap(csv_path, GRID_PLOT_OUTPUT_PATH)
    elif args.plot_only:
        csv_path = args.csv or latest_sweep_csv()
        plot_path = plot_sweep(csv_path, args.plot_out)
    else:
        base_config = load_config(BASE_CONFIG_PATH)
        rows = [collect_row(base_config, gamma, FIXED_W) for gamma in GAMMA_VALUES]
        csv_path = OUTPUTS_DIR / f"lindblad_gamma_sweep_{timestamp_token()}.csv"
        write_csv(rows, csv_path)
        print_summary(rows, csv_path)
        plot_path = plot_sweep(csv_path, args.plot_out)
    print("")
    print(f"plot_png: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
