"""Command-line entry point for Transition Grid Atlas.

Transition Grid Atlas is a reproducible research instrument for investigating
emergent transport laws. It is not a proof of a new law of physics. Its
responsibility is to:

1. simulate
2. validate
3. explore
4. persist results into an atlas
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, UTC
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from engine import LAB_RESULTS_FIELDS, run_transport_simulation
from engine.lab_modes import build_lab_config, run_lab_simulation
from explorer.parameter_sweep import detect_phase_boundary_zones, run_parameter_sweep
from explorer.phase_map import build_phase_matrix, save_phase_map_plot
from explorer.recursive_hunter import run_recursive_hunter
from hardware.constraints import (
    validate_coupling_range,
    validate_disorder_range,
    validate_gamma_range,
    validate_wafer_config,
)
from hardware.ensemble_statistics import (
    compute_bootstrap_summary,
    compute_bound_pressure_summary,
    compute_tradeoff_summary,
    compute_win_rates,
    plot_ensemble_statistics,
    read_ensemble_csv,
    write_statistics_json,
)
from hardware.lab_config_writer import write_lab_config_from_hardware_mapping
from hardware.motor_audit import run_transition_motor_bound_audit
from hardware.motor_ensemble import run_transition_motor_ensemble_study
from hardware.objectives import evaluate_motor_metrics, normalized_motor_objective
from hardware.motor_pareto import (
    plot_pareto_results,
    run_pareto_weight_sweep,
    transition_motor_pareto_from_dict,
    write_pareto_csv,
    write_pareto_summary_json,
)
from hardware.motor_pareto_audit import (
    run_pareto_audit,
    write_pareto_audit_csv,
    write_pareto_audit_summary_json,
    plot_pareto_audit,
)
from hardware.noise_controller import effective_gamma_from_controller, noise_controller_from_dict
from hardware.photonic_wafer import (
    disorder_strength_from_fabrication,
    photonic_wafer_from_dict,
)
from hardware.transition_tuner import (
    build_base_hamiltonian,
    fixed_grid_from_dict,
    noise_operators_from_profiles,
    random_transition_search,
    transition_tuner_config_from_dict,
)
from hardware.transition_motor import (
    build_default_motor_basis,
    build_control_hamiltonian,
    random_restart_transition_motor_search,
    transition_motor_config_from_dict,
)
from hardware.sensitivity import compute_sensitivity_matrix, write_sensitivity_csv
from hardware.wafer_ensemble import run_wafer_ensemble_study
from interface.visualiser import render_probability_animation
from inverse_transition_layer import run_inverse_transition_analysis
from validation.audit import build_audit_report
from validation.kta_audit import summarize_kta_audit
from validation.monte_carlo import summarise_samples


PROJECT_ROOT = Path(__file__).resolve().parent
ATLAS_RESULTS_DIR = PROJECT_ROOT / "atlas" / "results"
ATLAS_PLOTS_DIR = PROJECT_ROOT / "atlas" / "plots"
ATLAS_REPORTS_DIR = PROJECT_ROOT / "atlas" / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LAB_RESULTS_DIR = PROJECT_ROOT / "results"
LAB_TRAJECTORIES_DIR = LAB_RESULTS_DIR / "trajectories"
LAB_RENDERS_DIR = LAB_RESULTS_DIR / "renders"
LAB_LEDGER_PATH = LAB_RESULTS_DIR / "master_results.csv"
GENERATED_CONFIGS_DIR = PROJECT_ROOT / "configs" / "generated"
MASTER_RESULTS_PATH = ATLAS_RESULTS_DIR / "master_results.csv"
LATEST_REPORT_PATH = ATLAS_REPORTS_DIR / "latest_report.md"


def load_config(config_path: Path) -> dict[str, Any]:
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    base_path = PROJECT_ROOT / "config.yaml"
    with base_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    resolved = config_path.expanduser().resolve()
    if resolved == base_path.resolve():
        return base_config

    with resolved.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}
    return _deep_merge(base_config, override)


def ensure_output_dirs() -> None:
    ATLAS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ATLAS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ATLAS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LAB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LAB_TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    LAB_RENDERS_DIR.mkdir(parents=True, exist_ok=True)


def timestamp_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def lab_run_id(theory_mode: str) -> str:
    return f"{timestamp_token()}_{theory_mode}"


def append_lab_results(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_output_dirs()
    file_exists = LAB_LEDGER_PATH.exists()
    with LAB_LEDGER_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LAB_RESULTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LAB_RESULTS_FIELDS})


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_mapping_config_hash(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def resolve_lab_artifact_path(run_id: str | None, artifact_path: str | None) -> Path:
    if run_id and artifact_path:
        raise ValueError("Use either --run-id or --artifact-path, not both")
    if artifact_path:
        return Path(artifact_path).expanduser().resolve()
    if run_id:
        return (LAB_TRAJECTORIES_DIR / f"run_{run_id}.npz").resolve()
    raise ValueError("Either --run-id or --artifact-path is required")


def flatten_result(result: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "W": result["parameters"]["W"],
        "bias": result["parameters"]["bias"],
        "eta": result["parameters"]["eta"],
        "gamma": result["parameters"]["gamma"],
        "seed": result["parameters"]["seed"],
        "alpha": result["alpha"],
        "early_alpha": result["early_alpha"],
        "late_alpha": result["late_alpha"],
        "alpha_window_shift": result["alpha_window_shift"],
        "scaling_window_stable": result["scaling_window_stable"],
        "r2": result["r2"],
        "early_r2": result["early_r2"],
        "late_r2": result["late_r2"],
        "regression_rmse": result["regression_rmse"],
        "mean_current": result["mean_current"],
        "unitarity_error": result["unitarity_error"],
        "hermitian_error": result["hermitian_error"],
        "status": result["status"],
    }


def append_master_results(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_output_dirs()
    existing_rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    if MASTER_RESULTS_PATH.exists():
        with MASTER_RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                fieldnames.extend(reader.fieldnames)
            existing_rows.extend(reader)

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    normalized_rows = []
    for row in [*existing_rows, *rows]:
        normalized_rows.append({field: row.get(field, "") for field in fieldnames})

    with MASTER_RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latest_report(title: str, lines: list[str]) -> None:
    ensure_output_dirs()
    body = "\n".join(lines)
    LATEST_REPORT_PATH.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def is_physically_valid(result: dict[str, Any]) -> bool:
    return str(result["status"]).startswith("VALID")


def stability_rank_key(result: dict[str, Any]) -> tuple[float, ...]:
    return (
        0.0 if is_physically_valid(result) else 1.0,
        0.0 if result["scaling_window_stable"] else 1.0,
        -float(result["r2"]),
        float(result["alpha_window_shift"]),
        float(result["unitarity_error"]),
        float(result["hermitian_error"]),
        abs(float(result["alpha"]) - 0.50),
    )


def build_top_states_table(results: list[dict[str, Any]], limit: int = 5) -> list[str]:
    physical_states = sorted(
        [result for result in results if is_physically_valid(result)],
        key=stability_rank_key,
    )[:limit]

    lines = ["## Top stable physical states", ""]
    if not physical_states:
        physical_states = sorted(results, key=stability_rank_key)[:limit]
        lines.append(
            "No strictly valid physical states were found in this run. The table below lists the "
            "closest near-valid candidates so the ledger still captures the best observed regimes."
        )
        lines.append("")

    lines.extend(
        [
            "| Rank | Status | alpha | R² | window shift | W | bias | eta | gamma |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, result in enumerate(physical_states, start=1):
        params = result["parameters"]
        lines.append(
            "| "
            f"{index} | {result['status']} | {result['alpha']:.6f} | {result['r2']:.6f} | "
            f"{result['alpha_window_shift']:.6f} | {params['W']:.6f} | {params['bias']:.6f} | "
            f"{params['eta']:.6f} | {params['gamma']:.6f} |"
        )
    return lines


def latest_sweep_rows() -> list[dict[str, Any]]:
    sweep_files = sorted(ATLAS_RESULTS_DIR.glob("sweep_*.csv"))
    if not sweep_files:
        return []
    latest_file = sweep_files[-1]
    with latest_file.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def flatten_boundary_zone(zone: dict[str, float | str], source_batch: str) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_batch": source_batch,
        "mode": "sweep_boundary_zone",
        "W": zone["midpoint_W"],
        "bias": zone["bias"],
        "eta": zone["eta"],
        "gamma": zone["gamma"],
        "seed": "",
        "alpha": "",
        "early_alpha": "",
        "late_alpha": "",
        "alpha_window_shift": "",
        "scaling_window_stable": "",
        "r2": "",
        "early_r2": "",
        "late_r2": "",
        "regression_rmse": "",
        "mean_current": "",
        "unitarity_error": "",
        "hermitian_error": "",
        "status": "PHASE_BOUNDARY_ZONE",
        "boundary_left_W": zone["left_W"],
        "boundary_right_W": zone["right_W"],
        "boundary_left_alpha": zone["left_alpha"],
        "boundary_right_alpha": zone["right_alpha"],
        "boundary_gradient_magnitude": zone["gradient_magnitude"],
        "boundary_local_left_W": zone.get("local_gradient_left_W", ""),
        "boundary_local_right_W": zone.get("local_gradient_right_W", ""),
        "boundary_left_status": zone["left_status"],
        "boundary_right_status": zone["right_status"],
    }


def latest_boundary_zone_rows() -> list[dict[str, Any]]:
    if not MASTER_RESULTS_PATH.exists():
        return []
    with MASTER_RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "PHASE_BOUNDARY_ZONE"]
    if not rows:
        return []
    batches = sorted({row.get("source_batch", "") for row in rows if row.get("source_batch", "")})
    if not batches:
        return rows
    latest_batch = batches[-1]
    return [row for row in rows if row.get("source_batch") == latest_batch]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_phase_transition_summary(
    rows: list[dict[str, Any]],
    boundary_zone_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    lines = ["## Ballistic-to-localized boundary summary", ""]
    if boundary_zone_rows is None:
        boundary_zone_rows = latest_boundary_zone_rows()

    if boundary_zone_rows:
        lines.extend(
            [
                "Detected high-gradient transition slices from the latest sweep:",
                "",
                "| bias | eta | gamma | left W | right W | left alpha | right alpha | gradient | midpoint W |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in boundary_zone_rows:
            midpoint = np.float64(0.5) * (np.float64(row["boundary_left_W"]) + np.float64(row["boundary_right_W"]))
            lines.append(
                f"| {float(row['bias']):.6f} | {float(row['eta']):.6f} | {float(row['gamma']):.6f} | "
                f"{float(row['boundary_left_W']):.6f} | {float(row['boundary_right_W']):.6f} | "
                f"{float(row['boundary_left_alpha']):.6f} | {float(row['boundary_right_alpha']):.6f} | "
                f"{float(row['boundary_gradient_magnitude']):.6f} | {float(midpoint):.6f} |"
            )
        return lines

    if not rows:
        lines.append("No sweep data available yet, so no phase-boundary summary can be computed.")
        return lines

    grouped: dict[tuple[float, float, float], list[tuple[float, str]]] = {}
    for row in rows:
        key = (
            float(row["bias"]),
            float(row["eta"]),
            float(row["gamma"]),
        )
        grouped.setdefault(key, []).append((float(row["W"]), str(row["status"])))

    boundaries: list[tuple[float, float, float, float, float]] = []
    for (bias, eta, gamma), points in grouped.items():
        sorted_points = sorted(points, key=lambda item: item[0])
        for (prev_w, prev_status), (next_w, next_status) in zip(sorted_points, sorted_points[1:]):
            if prev_status == "VALID_BALLISTIC" and next_status == "VALID_LOCALIZED":
                boundaries.append((bias, eta, gamma, prev_w, next_w))
                break

    if not boundaries:
        lines.append(
            "No direct ballistic-to-localized boundary was observed in the latest sweep grid. "
            "The current grid may still contain intermediate weak-fit or diffusive-like regions."
        )
        return lines

    lines.extend(
        [
            "The latest sweep shows the following direct transitions as disorder increases along fixed "
            "bias/eta/gamma slices:",
            "",
            "| bias | eta | gamma | last ballistic W | first localized W | midpoint estimate |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bias, eta, gamma, prev_w, next_w in boundaries:
        midpoint = np.float64(0.5) * (np.float64(prev_w) + np.float64(next_w))
        lines.append(
            f"| {bias:.6f} | {eta:.6f} | {gamma:.6f} | {prev_w:.6f} | {next_w:.6f} | {midpoint:.6f} |"
        )
    return lines


def build_comprehensive_report(
    mode: str,
    headline_lines: list[str],
    results: list[dict[str, Any]],
    boundary_zones: list[dict[str, Any]] | None = None,
) -> list[str]:
    report_lines = [
        "## Run summary",
        "",
        *headline_lines,
        "",
        *build_top_states_table(results=results),
        "",
        *build_phase_transition_summary(latest_sweep_rows(), boundary_zone_rows=boundary_zones),
    ]
    return report_lines


def print_single_summary(result: dict[str, Any]) -> None:
    print(f"alpha: {result['alpha']:.6f}")
    print(f"R²: {result['r2']:.6f}")
    print(f"unitarity_error: {result['unitarity_error']:.3e}")
    print(f"hermitian_error: {result['hermitian_error']:.3e}")
    print(f"status: {result['status']}")


def build_inverse_report_lines(analysis: dict[str, Any]) -> list[str]:
    summary = analysis["summary"]
    lines = [
        "## Inverse transition summary",
        "",
        f"- timestamp_utc: {datetime.now(UTC).isoformat()}",
        "- mode: inverse",
        f"- generated_paths: {len(analysis['pairs'])}",
        f"- path_axis: {analysis['path_axis']}",
        f"- slice_keys: {list(analysis['slice_keys'])}",
        f"- dominance_threshold: {summary['dominance_threshold']:.3f}",
        f"- annihilation_threshold: {summary['annihilation_threshold']:.3f}",
        f"- symmetrical_paths: {summary['symmetrical_paths']}",
        f"- annihilated_paths: {summary['annihilated_paths']}",
        f"- dominant_paths: {summary['dominant_paths']}",
        f"- dominant_direction: {summary['dominant_direction']}",
        f"- average_annihilation_score: {summary['average_annihilation_score']:.6f}",
        f"- inverse_csv: {analysis['csv_output_path']}",
        f"- inverse_plot: {analysis['plot_output_path']}",
        "",
        "## Emergent regularities",
        "",
    ]

    reported_laws = summary["reported_laws"]
    if not reported_laws:
        lines.append("No directional regularities exceeded the configured dominance threshold.")
        return lines

    for index, law in enumerate(reported_laws[:10], start=1):
        lines.append(
            f"{index}. {law['summary']}. forward_path={law['forward_path']}. inverse_path={law['inverse_path']}."
        )
    return lines


def build_lab_report_lines(result: dict[str, Any], render_path: Path | None = None) -> list[str]:
    lab_cfg = result["config"]
    lines = [
        "## Experimental Quantum & RTT Lab summary",
        "",
        f"- timestamp_utc: {datetime.now(UTC).isoformat()}",
        f"- run_id: {result['run_id']}",
        f"- theory_mode: {lab_cfg.theory_mode}",
        f"- grid_size: {lab_cfg.grid_size}",
        f"- time_horizon_steps: {result['summary_row']['T']}",
        f"- dt: {lab_cfg.dt}",
        f"- W: {result['summary_row']['W']}",
        f"- gamma: {result['summary_row']['gamma']}",
        f"- eta: {lab_cfg.eta}",
        f"- x0: {lab_cfg.x0}",
        f"- sigma: {lab_cfg.sigma}",
        f"- k0: {lab_cfg.k0}",
        f"- alpha_late: {result['alpha_late']:.6f}",
        f"- trace_error: {result['trace_error']:.6e}",
        f"- ipr_final: {result['summary_row']['ipr_final']:.6f}",
        f"- edge_hit: {result['edge_hit']}",
        f"- falldown: {result['falldown']}",
        f"- unitarity_error: {result['unitarity_error']:.6e}",
        f"- hermitian_error: {result['hermitian_error']:.6e}",
        f"- trajectory_artifact: {result['artifact_path']}",
        f"- trajectory_artifact_hash: {result['artifact_hash']}",
        f"- config_hash: {result['config_hash']}",
    ]
    if render_path is not None:
        lines.append(f"- render_artifact: {render_path}")
    return lines


def run_lab_mode(
    config: dict[str, Any],
    *,
    mode: str | None = None,
    gamma: float | None = None,
    render: bool = False,
) -> int:
    ensure_output_dirs()
    lab_cfg = build_lab_config(config, mode_override=mode, gamma_override=gamma)
    run_id = lab_run_id(lab_cfg.theory_mode)
    result = run_lab_simulation(
        config=config,
        run_id=run_id,
        trajectories_dir=LAB_TRAJECTORIES_DIR,
        lab_config=lab_cfg,
    )

    audit_summary = summarize_kta_audit(
        probability_frames=result["probability_frames"],
        times=result["times"],
        trace_series=result["trace_series"],
        x0=lab_cfg.x0,
        config=lab_cfg,
        coherence_norm_final=result["summary_row"]["coherence_norm_final"],
    )
    result["trace_error"] = audit_summary["trace_error"]
    result["alpha_late"] = audit_summary["alpha_late"]
    result["edge_hit"] = audit_summary["edge_hit"]
    result["falldown"] = audit_summary["falldown_candidate"]
    result["summary_row"]["trace_error"] = audit_summary["trace_error"]
    result["summary_row"]["ipr_final"] = audit_summary["ipr_final"]
    result["summary_row"]["alpha_late"] = audit_summary["alpha_late"]
    result["summary_row"]["r2_late"] = audit_summary["r2_late"]
    result["summary_row"]["edge_hit"] = audit_summary["edge_hit"]
    result["summary_row"]["falldown_candidate"] = audit_summary["falldown_candidate"]
    result["summary_row"]["falldown_score"] = audit_summary["falldown_score"]
    result["summary_row"]["frame_corr_late"] = audit_summary["frame_corr_late"]
    result["summary_row"]["x_var_drift_late"] = audit_summary["x_var_drift_late"]
    result["summary_row"]["coherence_norm_final"] = audit_summary["coherence_norm_final"]
    result["summary_row"]["zeno_indicator"] = audit_summary["zeno_indicator"]
    result["summary_row"]["artifact_hash"] = artifact_sha256(result["artifact_path"])
    result["artifact_hash"] = result["summary_row"]["artifact_hash"]
    append_lab_results([result["summary_row"]])

    render_path = None
    should_render = bool(render or lab_cfg.render_animate)
    if should_render:
        render_path = LAB_RENDERS_DIR / f"{run_id}.mp4"
        render_probability_animation(artifact_path=result["artifact_path"], output_path=render_path, fps=lab_cfg.render_fps)

    report_lines = build_lab_report_lines(result, render_path=render_path)
    write_latest_report("Latest Transition Grid Atlas report", report_lines)

    print(f"run_id: {run_id}")
    print(f"theory_mode: {lab_cfg.theory_mode}")
    print(f"trajectory_artifact: {result['artifact_path']}")
    print(f"artifact_hash: {result['artifact_hash']}")
    print(f"trace_error: {result['trace_error']:.6e}")
    print(f"ipr_final: {result['summary_row']['ipr_final']:.6f}")
    print(f"alpha_late: {result['alpha_late']:.6f}")
    print(f"edge_hit: {result['edge_hit']}")
    print(f"falldown: {result['falldown']}")
    print(f"config_hash: {result['config_hash']}")
    if render_path is not None:
        print(f"render_artifact: {render_path}")
    return 0


def run_animate_mode(*, trajectory: str, output_path: str) -> int:
    ensure_output_dirs()
    artifact = Path(trajectory).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    render_probability_animation(artifact_path=artifact, output_path=output, fps=30)
    print(f"render_artifact: {output}")
    return 0


def run_single_mode(config: dict[str, Any]) -> int:
    result = run_transport_simulation(config=config)
    append_master_results([flatten_result(result, mode="single")])
    write_latest_report(
        title="Latest Transition Grid Atlas report",
        lines=build_comprehensive_report(
            mode="single",
            headline_lines=[
            f"- timestamp_utc: {datetime.now(UTC).isoformat()}",
            f"- mode: single",
            f"- parameters: {result['parameters']}",
            f"- alpha: {result['alpha']:.6f}",
            f"- early_alpha: {result['early_alpha']:.6f}",
            f"- late_alpha: {result['late_alpha']:.6f}",
            f"- alpha_window_shift: {result['alpha_window_shift']:.6f}",
            f"- scaling_window_stable: {result['scaling_window_stable']}",
            f"- R²: {result['r2']:.6f}",
            f"- unitarity_error: {result['unitarity_error']:.3e}",
            f"- hermitian_error: {result['hermitian_error']:.3e}",
            f"- status: {result['status']}",
        ],
            results=[result],
        ),
    )
    print_single_summary(result)
    return 0


def run_monte_carlo_mode(config: dict[str, Any], runs: int | None) -> int:
    ensure_output_dirs()
    mc_cfg = config["montecarlo"]
    total_runs = int(runs or mc_cfg["runs"])
    base_seed = int(mc_cfg["base_seed"])
    vary_cfg = mc_cfg["vary"]
    base_params = dict(config["parameters"])
    rng = np.random.default_rng(base_seed)

    results = []
    flat_rows = []
    for run_index in range(total_runs):
        sampled_parameters = {
            "W": max(0.0, float(base_params["W"] + rng.normal(0.0, vary_cfg["W_std"]))),
            "bias": float(base_params["bias"] + rng.normal(0.0, vary_cfg["bias_std"])),
            "eta": max(0.0, float(base_params["eta"] + rng.normal(0.0, vary_cfg["eta_std"]))),
            "gamma": max(0.0, float(base_params["gamma"] + rng.normal(0.0, vary_cfg["gamma_std"]))),
            "seed": base_seed + run_index,
        }
        result = run_transport_simulation(config=config, parameters=sampled_parameters)
        results.append(result)
        flat_rows.append(flatten_result(result, mode="montecarlo"))

    alpha_summary = summarise_samples(np.array([result["alpha"] for result in results], dtype=np.float64))
    r2_summary = summarise_samples(np.array([result["r2"] for result in results], dtype=np.float64))

    stamp = timestamp_token()
    csv_path = ATLAS_RESULTS_DIR / f"montecarlo_{stamp}.csv"
    write_rows_csv(csv_path, flat_rows)
    append_master_results(flat_rows)

    for metric_key, label in (("alpha", "Alpha"), ("r2", "R²")):
        values = np.array([result[metric_key] for result in results], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=18, color="#1f77b4", edgecolor="white")
        ax.set_title(f"Monte Carlo {label} distribution")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(ATLAS_PLOTS_DIR / f"montecarlo_{metric_key}_{stamp}.png", dpi=180)
        plt.close(fig)

    write_latest_report(
        title="Latest Transition Grid Atlas report",
        lines=build_comprehensive_report(
            mode="montecarlo",
            headline_lines=[
            f"- timestamp_utc: {datetime.now(UTC).isoformat()}",
            f"- mode: montecarlo",
            f"- runs: {total_runs}",
            f"- mean alpha: {alpha_summary['mean']:.6f}",
            f"- std alpha: {alpha_summary['std']:.6f}",
            f"- mean R²: {r2_summary['mean']:.6f}",
            f"- alpha ci95: [{alpha_summary['ci95_low']:.6f}, {alpha_summary['ci95_high']:.6f}]",
            f"- R² ci95: [{r2_summary['ci95_low']:.6f}, {r2_summary['ci95_high']:.6f}]",
            f"- results_csv: {csv_path}",
        ],
            results=results,
        ),
    )

    print(f"mean alpha: {alpha_summary['mean']:.6f}")
    print(f"std alpha: {alpha_summary['std']:.6f}")
    print(f"mean R²: {r2_summary['mean']:.6f}")
    print(f"results_csv: {csv_path}")
    return 0


def run_sweep_mode(config: dict[str, Any]) -> int:
    ensure_output_dirs()
    results = run_parameter_sweep(config=config)
    boundary_zones = detect_phase_boundary_zones(config=config, results=results)
    flat_rows = [flatten_result(result, mode="sweep") for result in results]
    stamp = timestamp_token()
    csv_path = ATLAS_RESULTS_DIR / f"sweep_{stamp}.csv"
    write_rows_csv(csv_path, flat_rows)
    boundary_rows = [flatten_boundary_zone(zone, source_batch=stamp) for zone in boundary_zones]
    append_master_results([*flat_rows, *boundary_rows])

    sweep_cfg = config["sweep"]
    x_key, y_key = sweep_cfg["phase_map_axes"]
    matrix, x_values, y_values = build_phase_matrix(
        results=results,
        x_key=x_key,
        y_key=y_key,
        metric_key=sweep_cfg["phase_metric"],
        valid_only=bool(sweep_cfg["phase_valid_only"]),
    )
    matrix_csv_path = ATLAS_RESULTS_DIR / f"phase_map_{x_key}_{y_key}_{stamp}.csv"
    with matrix_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{y_key}\\{x_key}", *[f"{value:.6f}" for value in x_values]])
        for y_value, row in zip(y_values, matrix):
            writer.writerow([f"{y_value:.6f}", *[f"{value:.6f}" if not np.isnan(value) else "nan" for value in row]])

    plot_path = ATLAS_PLOTS_DIR / f"phase_map_{x_key}_{y_key}_{stamp}.png"
    save_phase_map_plot(
        matrix=matrix,
        x_values=x_values,
        y_values=y_values,
        x_label=x_key,
        y_label=y_key,
        title=f"Phase map: {sweep_cfg['phase_metric']} over {x_key} × {y_key}",
        colorbar_label=sweep_cfg["phase_metric"],
        output_path=plot_path,
    )

    valid_fraction = float(np.mean([result["status"].startswith("VALID") for result in results], dtype=np.float64))
    write_latest_report(
        title="Latest Transition Grid Atlas report",
        lines=build_comprehensive_report(
            mode="sweep",
            headline_lines=[
            f"- timestamp_utc: {datetime.now(UTC).isoformat()}",
            f"- mode: sweep",
            f"- combinations: {len(results)}",
            f"- valid_fraction: {valid_fraction:.6f}",
            f"- boundary_zone_count: {len(boundary_zones)}",
            f"- sweep_csv: {csv_path}",
            f"- phase_matrix_csv: {matrix_csv_path}",
            f"- phase_plot: {plot_path}",
        ],
            results=results,
            boundary_zones=boundary_rows,
        ),
    )

    print(f"sweep combinations: {len(results)}")
    print(f"boundary zones detected: {len(boundary_zones)}")
    print(f"phase map saved to: {plot_path}")
    print(f"matrix csv saved to: {matrix_csv_path}")
    return 0


def run_hunter_mode(config: dict[str, Any]) -> int:
    ensure_output_dirs()
    boundary_zone_rows = latest_boundary_zone_rows()
    hunt = run_recursive_hunter(config=config, boundary_zones=boundary_zone_rows)
    best_result = hunt["best_result"]
    all_results = hunt["all_results"]
    boundary_anchor = hunt["boundary_anchor"]
    flat_rows = []
    for result in all_results:
        row = flatten_result(result, mode="hunter")
        row["round"] = result["round"]
        row["score"] = result["score"]
        flat_rows.append(row)

    stamp = timestamp_token()
    csv_path = ATLAS_RESULTS_DIR / f"hunter_{stamp}.csv"
    write_rows_csv(csv_path, flat_rows)
    append_master_results([flatten_result(best_result, mode="hunter_best")])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        [result["alpha"] for result in all_results],
        [result["r2"] for result in all_results],
        c=[result["round"] for result in all_results],
        cmap="plasma",
        s=36,
    )
    ax.axvline(config["hunter"]["target_alpha"], color="black", linestyle="--", linewidth=1.0)
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("alpha")
    ax.set_ylabel("R²")
    ax.set_title("Recursive hunter candidate cloud")
    fig.tight_layout()
    plot_path = ATLAS_PLOTS_DIR / f"hunter_candidates_{stamp}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    write_latest_report(
        title="Latest Transition Grid Atlas report",
        lines=build_comprehensive_report(
            mode="hunter",
            headline_lines=[
            f"- timestamp_utc: {datetime.now(UTC).isoformat()}",
            f"- mode: hunter",
            f"- boundary_zone_source_count: {len(boundary_zone_rows)}",
            f"- boundary_anchor: {boundary_anchor}",
            f"- best parameters: {best_result['parameters']}",
            f"- best alpha: {best_result['alpha']:.6f}",
            f"- best R²: {best_result['r2']:.6f}",
            f"- best early_alpha: {best_result['early_alpha']:.6f}",
            f"- best late_alpha: {best_result['late_alpha']:.6f}",
            f"- best alpha_window_shift: {best_result['alpha_window_shift']:.6f}",
            f"- best unitarity_error: {best_result['unitarity_error']:.3e}",
            f"- best hermitian_error: {best_result['hermitian_error']:.3e}",
            f"- best status: {best_result['status']}",
            f"- candidate_csv: {csv_path}",
            f"- candidate_plot: {plot_path}",
        ],
            results=all_results,
            boundary_zones=boundary_zone_rows,
        ),
    )

    print(f"best alpha: {best_result['alpha']:.6f}")
    print(f"best R²: {best_result['r2']:.6f}")
    print(f"best unitarity_error: {best_result['unitarity_error']:.3e}")
    print(f"best hermitian_error: {best_result['hermitian_error']:.3e}")
    print(f"status: {best_result['status']}")
    return 0


def run_audit_mode(config: dict[str, Any]) -> int:
    report = build_audit_report(config=config, project_root=PROJECT_ROOT)
    print(report)
    return 0


def run_inverse_mode(config: dict[str, Any]) -> int:
    ensure_output_dirs()
    csv_path = OUTPUTS_DIR / "inverse_analysis.csv"
    plot_path = OUTPUTS_DIR / "inverse_transition_map.png"
    analysis = run_inverse_transition_analysis(
        config=config,
        csv_output_path=csv_path,
        plot_output_path=plot_path,
    )

    write_latest_report(
        title="Latest Transition Grid Atlas report",
        lines=build_inverse_report_lines(analysis),
    )

    summary = analysis["summary"]
    print(f"generated paths: {len(analysis['pairs'])}")
    print(f"symmetrical paths: {summary['symmetrical_paths']}")
    print(f"annihilated paths: {summary['annihilated_paths']}")
    print(f"dominant paths: {summary['dominant_paths']}")
    print(f"dominant direction: {summary['dominant_direction']}")
    print(f"average annihilation score: {summary['average_annihilation_score']:.6f}")
    print(f"inverse csv: {csv_path}")
    print(f"inverse plot: {plot_path}")
    return 0


def run_hardware_map_mode(
    config: dict[str, Any],
    *,
    emit_lab_config: bool = False,
    out_config: str | None = None,
) -> int:
    hardware_block = dict(config.get("hardware", {}))
    noise_block = dict(config.get("noise_controller", {}))
    mapping_block = dict(config.get("mapping", {}))
    if not hardware_block:
        raise ValueError("hardware-map requires a hardware block in the config")
    if not noise_block:
        raise ValueError("hardware-map requires a noise_controller block in the config")

    wafer = photonic_wafer_from_dict(hardware_block)
    controller = noise_controller_from_dict(noise_block)
    validate_wafer_config(wafer)
    validate_coupling_range(wafer.coupling_j)

    W_eff = disorder_strength_from_fabrication(wafer)
    gamma_eff = effective_gamma_from_controller(controller)
    validate_disorder_range(W_eff)
    validate_gamma_range(gamma_eff)

    target_indices = [int(index) for index in mapping_block.get("target_indices", [])]
    for index in target_indices:
        if index < 0 or index >= wafer.n_sites:
            raise ValueError(f"target index out of range for wafer layout: {index}")

    generated_lab_config = None
    run_command = None
    if emit_lab_config:
        config_hash = hardware_mapping_config_hash(
            {
                "hardware": hardware_block,
                "noise_controller": noise_block,
                "mapping": mapping_block,
                "W_eff": round(W_eff, 12),
                "gamma_eff": round(gamma_eff, 12),
            }
        )
        output_path = (
            Path(out_config).expanduser().resolve()
            if out_config
            else (GENERATED_CONFIGS_DIR / f"photonic_wafer_{config_hash}.yaml").resolve()
        )
        metadata = {
            "layout": wafer.layout,
            "n_sites": wafer.n_sites,
            "calibration_note": str(
                mapping_block.get(
                    "calibration_note",
                    "Phenomenological first-pass mapping from wafer disorder and stochastic phase modulation to KTA W and gamma.",
                )
            ),
        }
        generated_lab_config = write_lab_config_from_hardware_mapping(
            output_path,
            W_eff=W_eff,
            gamma_eff=gamma_eff,
            n_sites=wafer.n_sites,
            target_indices=target_indices,
            metadata=metadata,
        )
        run_command = f"python run.py --config {generated_lab_config} lab --mode lindblad"
    else:
        run_command = (
            f"python run.py lab --mode lindblad --gamma {gamma_eff:.6f} "
            f"# set effective W={W_eff:.6f} in a lab override config"
        )

    print("Photonic wafer mapping")
    print(f"layout = {wafer.layout}")
    print(f"n_sites = {wafer.n_sites}")
    print(f"W_eff = {W_eff:.6f}")
    print(f"gamma_eff = {gamma_eff:.6f}")
    print(f"target_detectors = {target_indices}")
    if generated_lab_config is not None:
        print(f"generated_lab_config = {generated_lab_config}")
    print(f"recommended_kta_command = {run_command}")
    print(f"run_command = {run_command}")
    return 0


def run_transition_tune_mode(config: dict[str, Any]) -> int:
    tuner_block = dict(config.get("transition_tuner", {}))
    if not tuner_block:
        raise ValueError("transition-tune requires a transition_tuner block in the config")

    grid = fixed_grid_from_dict(dict(tuner_block.get("grid", {})))
    noise_block = dict(tuner_block.get("noise", {}))
    search_block = dict(tuner_block.get("search", {}))
    if not noise_block.get("profiles"):
        raise ValueError("transition-tune requires noise profiles in transition_tuner.noise.profiles")

    noise_ops = noise_operators_from_profiles(list(noise_block["profiles"]))
    tuner_config = transition_tuner_config_from_dict(search_block)
    result = random_transition_search(grid, noise_ops, tuner_config)
    improvement = result.best_objective - result.baseline_objective

    print("Transition dynamics tuner")
    print("subspace_definition = dynamic_transport_modes_from_H")
    print("noise_overlap_control_dependent = true")
    print(f"fixed_grid_sites = {grid.n_sites}")
    print(f"fixed_grid_edges = {len(grid.edges)}")
    print(f"baseline_transport_efficiency = {result.baseline_transport_efficiency:.6f}")
    print(f"best_transport_efficiency = {result.best_transport_efficiency:.6f}")
    print(f"baseline_noise_overlap = {result.baseline_noise_overlap:.6f}")
    print(f"best_noise_overlap = {result.best_noise_overlap:.6f}")
    print(f"baseline_suppression_score = {result.baseline_suppression_score:.6f}")
    print(f"best_suppression_score = {result.best_suppression_score:.6f}")
    print(f"baseline_objective = {result.baseline_objective:.6f}")
    print(f"best_objective = {result.best_objective:.6f}")
    print(f"improvement = {improvement:.6f}")
    print(
        "interpretation = This is an algebraic first-pass tuner for hardware-native error suppression. "
        "It does not implement full QEC, syndrome extraction or recovery. "
        "It searches transition parameters on a fixed grid that reduce phase-noise overlap "
        "with information-carrying modes."
    )
    return 0


def run_wafer_ensemble_mode(config: dict[str, Any]) -> int:
    _, summary, csv_path, summary_path = run_wafer_ensemble_study(config)
    print("Synthetic wafer ensemble study")
    print(f"n_samples = {summary.n_samples}")
    print(f"success_rate = {summary.success_rate:.6f}")
    print(f"mean_transport_gain = {summary.mean_transport_gain:.6f}")
    print(f"mean_noise_overlap_reduction = {summary.mean_noise_overlap_reduction:.6f}")
    print(f"mean_objective_gain = {summary.mean_objective_gain:.6f}")
    print(f"mean_baseline_transport_efficiency = {summary.mean_baseline_transport_efficiency:.6f}")
    print(f"mean_best_transport_efficiency = {summary.mean_best_transport_efficiency:.6f}")
    print(f"mean_baseline_noise_overlap = {summary.mean_baseline_noise_overlap:.6f}")
    print(f"mean_best_noise_overlap = {summary.mean_best_noise_overlap:.6f}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print(
        "interpretation = This synthetic ensemble tests whether transition-dynamics tuning remains beneficial "
        "under sampled fabrication disorder and phase-noise profiles. It is a hardware-native "
        "error-suppression study, not full QEC."
    )
    return 0


def run_transition_motor_mode(
    config: dict[str, Any],
    *,
    report_sensitivity: bool = False,
    operating_mode: str | None = None,
) -> int:
    if operating_mode:
        config = dict(config)
        block = dict(config["transition_motor"])
        objective_block = dict(block.get("objective", {}))
        objective_block["selected_mode"] = operating_mode
        block["objective"] = objective_block
        config["transition_motor"] = block
    motor_config, outputs = transition_motor_config_from_dict(config)
    grid = motor_config.grid
    registry = motor_config.knob_registry
    basis = build_default_motor_basis(grid)
    H0 = build_base_hamiltonian(grid)
    result = random_restart_transition_motor_search(
        H0,
        grid,
        motor_config.noise_profiles,
        basis,
        registry,
        motor_config,
    )

    sensitivity_path = None
    if report_sensitivity or outputs.get("sensitivity_csv_path"):
        times = np.linspace(motor_config.time_min, motor_config.time_max, motor_config.n_time_samples, dtype=np.float64)
        baseline_H = build_control_hamiltonian(H0, grid, result.baseline_theta, basis, registry)
        baseline_raw_metrics = evaluate_motor_metrics(
            baseline_H,
            motor_config.noise_profiles,
            grid,
            result.baseline_theta,
            times,
            motor_config.objective_weights,
            n_modes=motor_config.n_transport_modes,
        )

        def metrics_fn(theta: dict[str, float]):
            H = build_control_hamiltonian(H0, grid, theta, basis, registry)
            from dataclasses import replace

            raw_metrics = evaluate_motor_metrics(
                H,
                motor_config.noise_profiles,
                grid,
                theta,
                times,
                motor_config.objective_weights,
                n_modes=motor_config.n_transport_modes,
            )
            if motor_config.objective_mode not in {"normalized", "calibrated"}:
                return raw_metrics
            objective = normalized_motor_objective(
                baseline_metrics=baseline_raw_metrics,
                candidate_metrics=raw_metrics,
                weights=motor_config.objective_weights,
                normalization_scales=motor_config.normalization_scales,
            )
            return replace(raw_metrics, objective=objective)

        entries = compute_sensitivity_matrix(
            metrics_fn,
            result.best_theta,
            registry,
            eps=motor_config.finite_diff_eps,
            metric_names=[
                "transport_efficiency",
                "noise_action_on_info",
                "noise_leakage",
                "control_cost",
                "objective",
            ],
        )
        sensitivity_path = write_sensitivity_csv(
            outputs.get("sensitivity_csv_path", OUTPUTS_DIR / "transition_motor_sensitivity.csv"),
            entries,
        )

    print("Transition Motor Instrumentation")
    print(f"fixed_grid_sites = {grid.n_sites}")
    print(f"fixed_grid_edges = {len(grid.edges)}")
    print(f"active_knobs = {registry.names()}")
    print(f"objective_mode = {motor_config.objective_mode}")
    print(f"selected_objective_mode = {motor_config.selected_objective_mode or 'legacy_objective_weights'}")
    if motor_config.objective_mode_registry is not None and motor_config.selected_objective_mode is not None:
        print(
            "selected_objective_mode_description = "
            f"{motor_config.objective_mode_registry.get(motor_config.selected_objective_mode).description}"
        )
    print(
        "available_objective_modes = "
        f"{motor_config.objective_mode_registry.names() if motor_config.objective_mode_registry is not None else ['legacy_objective_weights']}"
    )
    print(f"objective_weights = {motor_config.objective_weights}")
    if motor_config.normalization_scales is not None:
        print(
            "normalization_scales = "
            f"{{'transport': {motor_config.normalization_scales.transport:.6f}, "
            f"'noise_action': {motor_config.normalization_scales.noise_action:.6f}, "
            f"'leakage': {motor_config.normalization_scales.leakage:.6f}, "
            f"'control_cost': {motor_config.normalization_scales.control_cost:.6f}}}"
        )
    print(f"baseline_transport_efficiency = {result.baseline_metrics.transport_efficiency:.6f}")
    print(f"best_transport_efficiency = {result.best_metrics.transport_efficiency:.6f}")
    print(f"baseline_noise_action_on_info = {result.baseline_metrics.noise_action_on_info:.6f}")
    print(f"best_noise_action_on_info = {result.best_metrics.noise_action_on_info:.6f}")
    print(f"baseline_noise_leakage = {result.baseline_metrics.noise_leakage:.6f}")
    print(f"best_noise_leakage = {result.best_metrics.noise_leakage:.6f}")
    print(f"baseline_control_cost = {result.baseline_metrics.control_cost:.6f}")
    print(f"best_control_cost = {result.best_metrics.control_cost:.6f}")
    print(f"baseline_objective = {result.baseline_metrics.objective:.6f}")
    print(f"best_objective = {result.best_metrics.objective:.6f}")
    print(f"objective_improvement = {result.objective_improvement:.6f}")
    print(f"best_theta = {result.best_theta}")
    print(f"top_sensitivities = {result.top_sensitivities}")
    print(f"sensitivity_csv_path = {sensitivity_path if sensitivity_path is not None else 'not_written'}")
    print(
        "interpretation = The transition motor does not change the physical grid. "
        "It exposes interpretable knobs on the effective Hamiltonian and measures "
        "how each knob affects transport, noise-action on information modes, leakage "
        "and control cost. This is hardware-native error suppression, not full QEC."
    )
    return 0


def run_transition_motor_audit_mode(config: dict[str, Any]) -> int:
    motor_config, _ = transition_motor_config_from_dict(config)
    audit = run_transition_motor_bound_audit(motor_config)

    statuses = audit.bound_statuses
    ablations = sorted(audit.ablations, key=lambda item: item.objective_loss_from_ablation, reverse=True)
    efficiencies = sorted(audit.efficiencies, key=lambda item: item.gain_per_cost, reverse=True)
    upper = [item.name for item in statuses if item.at_upper_bound]
    lower = [item.name for item in statuses if item.at_lower_bound]
    near = [item.name for item in statuses if item.near_bound]

    print("Transition Motor Bound-Pressure Audit")
    print(f"saturated_knobs = {sum(1 for item in statuses if item.at_lower_bound or item.at_upper_bound)}")
    print(f"near_bound_knobs = {near}")
    print(f"knobs_at_upper_bound = {upper}")
    print(f"knobs_at_lower_bound = {lower}")
    print(
        "top_knobs_by_objective_loss = "
        f"{[{'knob': item.knob, 'objective_loss_from_ablation': item.objective_loss_from_ablation} for item in ablations[:3]]}"
    )
    print(
        "top_knobs_by_gain_per_cost = "
        f"{[{'knob': item.knob, 'gain_per_cost': item.gain_per_cost} for item in efficiencies[:3]]}"
    )
    for item in audit.limit_sweep:
        print(f"limit_scale_{item.scale:.2f}_objective_improvement = {item.objective_improvement:.6f}")
    print(f"mean_best_objective = {audit.seed_summary['mean_best_objective']:.6f}")
    print(f"std_best_objective = {audit.seed_summary['std_best_objective']:.6f}")
    print(f"min_best_objective = {audit.seed_summary['min_best_objective']:.6f}")
    print(f"max_best_objective = {audit.seed_summary['max_best_objective']:.6f}")
    print(f"constraint_limited = {str(audit.constraint_limited).lower()}")
    print(f"recommended_action = {audit.recommended_action}")
    return 0


def run_transition_motor_ensemble_mode(config: dict[str, Any]) -> int:
    _, summary, csv_path, summary_path = run_transition_motor_ensemble_study(config)
    print("Transition Motor Ensemble Study")
    print(f"n_samples = {summary.n_samples}")
    print(f"success_rate = {summary.success_rate:.6f}")
    print(f"mean_detector_success_gain = {summary.mean_detector_success_gain:.6f}")
    print(f"median_detector_success_gain = {summary.median_detector_success_gain:.6f}")
    print(f"mean_transport_gain = {summary.mean_transport_gain:.6f}")
    print(f"mean_noise_action_reduction = {summary.mean_noise_action_reduction:.6f}")
    print(f"mean_noise_leakage_reduction = {summary.mean_noise_leakage_reduction:.6f}")
    print(f"mean_objective_gain = {summary.mean_objective_gain:.6f}")
    print(f"mean_saturated_knobs = {summary.mean_saturated_knobs:.6f}")
    print(f"mean_near_bound_knobs = {summary.mean_near_bound_knobs:.6f}")
    print(f"mean_baseline_detector_success = {summary.mean_baseline_detector_success:.6f}")
    print(f"mean_best_detector_success = {summary.mean_best_detector_success:.6f}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print(
        "interpretation = This synthetic ensemble tests whether transition-motor controls improve "
        "detector-output success and reduce noise-action under sampled wafer disorder and phase-noise "
        "profiles. It is hardware-native error suppression, not full QEC or experimental validation."
    )
    return 0


def run_transition_motor_ensemble_stats_mode(
    *,
    csv_path: str,
    out_json: str,
    out_plot: str,
) -> int:
    rows = read_ensemble_csv(csv_path)
    win_rates = compute_win_rates(rows)
    bootstrap = compute_bootstrap_summary(
        rows,
        [
            "detector_success_gain",
            "noise_action_reduction",
            "noise_leakage_reduction",
            "objective_gain",
        ],
    )
    bound_pressure = compute_bound_pressure_summary(rows)
    tradeoffs = compute_tradeoff_summary(rows)
    json_path = write_statistics_json(
        out_json,
        win_rates=win_rates,
        bootstrap=bootstrap,
        bound_pressure=bound_pressure,
        tradeoffs=tradeoffs,
    )
    plot_path = plot_ensemble_statistics(rows, out_plot)

    bootstrap_by_metric = {item.metric: item for item in bootstrap}
    detector_boot = bootstrap_by_metric["detector_success_gain"]
    noise_boot = bootstrap_by_metric["noise_action_reduction"]
    leakage_boot = bootstrap_by_metric["noise_leakage_reduction"]

    print("Transition Motor Ensemble Statistics")
    print(f"n_samples = {win_rates.n_samples}")
    print(f"detector_win_rate = {win_rates.detector_win_rate:.6f}")
    print(f"noise_action_win_rate = {win_rates.noise_action_win_rate:.6f}")
    print(f"leakage_win_rate = {win_rates.leakage_win_rate:.6f}")
    print(f"objective_win_rate = {win_rates.objective_win_rate:.6f}")
    print(f"detector_and_noise_win_rate = {win_rates.detector_and_noise_win_rate:.6f}")
    print(f"all_core_metrics_win_rate = {win_rates.all_core_metrics_win_rate:.6f}")
    print(f"bootstrap_detector_gain_mean = {detector_boot.mean:.6f}")
    print(f"bootstrap_detector_gain_95ci = ({detector_boot.ci_low:.6f}, {detector_boot.ci_high:.6f})")
    print(f"bootstrap_noise_action_reduction_mean = {noise_boot.mean:.6f}")
    print(f"bootstrap_noise_action_reduction_95ci = ({noise_boot.ci_low:.6f}, {noise_boot.ci_high:.6f})")
    print(f"bootstrap_leakage_reduction_mean = {leakage_boot.mean:.6f}")
    print(f"bootstrap_leakage_reduction_95ci = ({leakage_boot.ci_low:.6f}, {leakage_boot.ci_high:.6f})")
    print(f"mean_saturated_knobs = {bound_pressure.mean_saturated_knobs:.6f}")
    print(f"detector_vs_noise_corr = {tradeoffs.detector_vs_noise_corr:.6f}")
    print(f"stats_json_path = {json_path}")
    print(f"stats_plot_path = {plot_path}")
    print(
        "interpretation = This statistical audit checks whether the synthetic ensemble improvement "
        "is distributed across samples or dominated by outliers. It supports detector-output robustness "
        "analysis, not experimental validation or full QEC."
    )
    return 0


def run_transition_motor_pareto_mode(config: dict[str, Any]) -> int:
    results, summary = run_pareto_weight_sweep(config)
    parsed = transition_motor_pareto_from_dict(config)
    csv_path = write_pareto_csv(parsed.outputs["csv_path"], results)
    summary_path = write_pareto_summary_json(parsed.outputs["summary_path"], results, summary)
    plot_path = plot_pareto_results(results, parsed.outputs["plot_path"])

    print("Transition Motor Pareto Sweep")
    print(f"n_weight_sets = {summary.n_weight_sets}")
    print(f"n_samples_per_weight_set = {results[0].n_samples if results else 0}")
    print(f"pareto_optimal_names = {summary.pareto_optimal_names}")
    print(f"best_detector_name = {summary.best_detector_name}")
    print(f"best_noise_action_name = {summary.best_noise_action_name}")
    print(f"best_leakage_name = {summary.best_leakage_name}")
    print(f"best_balanced_name = {summary.best_balanced_name}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print(f"plot_path = {plot_path}")
    for result in results:
        print(
            f"{result.name} = "
            f"{{'success_rate': {result.success_rate:.6f}, "
            f"'mean_detector_success_gain': {result.mean_detector_success_gain:.6f}, "
            f"'mean_noise_action_reduction': {result.mean_noise_action_reduction:.6f}, "
            f"'mean_noise_leakage_reduction': {result.mean_noise_leakage_reduction:.6f}, "
            f"'mean_best_control_cost': {result.mean_best_control_cost:.6f}, "
            f"'mean_saturated_knobs': {result.mean_saturated_knobs:.6f}, "
            f"'pareto_optimal': {str(result.is_pareto_optimal).lower()}}}"
        )
    print(
        "interpretation = This Pareto sweep tests how objective weights shift the transition motor "
        "between detector-output optimization, noise-action reduction, leakage control and actuator cost. "
        "It is synthetic ensemble analysis, not experimental validation or full QEC."
    )
    return 0


def run_transition_motor_pareto_audit_mode(config_path: Path) -> int:
    audit_result = run_pareto_audit(config_path)
    csv_path = write_pareto_audit_csv(str(audit_result.csv_path), audit_result)
    summary_path = write_pareto_audit_summary_json(str(audit_result.summary_path), audit_result)
    plot_path = plot_pareto_audit(audit_result, str(audit_result.plot_path))

    detector_ci = {
        item.preset_name: (item.ci_low, item.ci_high)
        for item in audit_result.confidence_intervals
        if item.metric == "detector_success_gain"
    }
    noise_ci = {
        item.preset_name: (item.ci_low, item.ci_high)
        for item in audit_result.confidence_intervals
        if item.metric == "noise_action_reduction"
    }
    leakage_ci = {
        item.preset_name: (item.ci_low, item.ci_high)
        for item in audit_result.confidence_intervals
        if item.metric == "noise_leakage_reduction"
    }
    component_lookup = {item.preset_name: item for item in audit_result.component_scales}
    knob_lookup = {item.preset_name: item for item in audit_result.knob_profiles}

    print("Transition Motor Pareto Audit")
    print(f"n_presets = {audit_result.summary.n_presets}")
    print(f"base_preset_names = {audit_result.summary.base_preset_names}")
    print(f"stress_preset_names = {audit_result.summary.stress_preset_names}")
    print(f"dominant_component_overall = {audit_result.summary.dominant_component_overall}")
    print(f"component_scaling_issue = {str(audit_result.summary.component_scaling_issue).lower()}")
    print(f"mean_pairwise_regime_distance = {audit_result.summary.mean_pairwise_regime_distance:.6f}")
    print(f"closest_presets = {audit_result.summary.closest_presets}")
    print(f"most_separated_presets = {audit_result.summary.most_separated_presets}")
    print(f"compressed_frontier = {str(audit_result.summary.compressed_frontier).lower()}")
    print(f"best_normalized_score_name = {audit_result.best_normalized_score_name}")
    print(f"csv_path = {csv_path}")
    print(f"summary_path = {summary_path}")
    print(f"plot_path = {plot_path}")
    print("Component scale:")
    for result in audit_result.pareto_results:
        component = component_lookup[result.name]
        print(
            f"{result.name} = {{'dominant_component': '{component.dominant_component}', "
            f"'detector_to_noise_ratio': {component.detector_to_noise_ratio:.6f}, "
            f"'detector_to_leakage_ratio': {component.detector_to_leakage_ratio:.6f}}}"
        )
    print("Confidence intervals:")
    print(f"detector_gain_95ci = {detector_ci}")
    print(f"noise_action_reduction_95ci = {noise_ci}")
    print(f"leakage_reduction_95ci = {leakage_ci}")
    print("Knob profiles:")
    for result in audit_result.pareto_results:
        profile = knob_lookup[result.name]
        dominant_knobs = sorted(profile.mean_theta, key=lambda name: abs(profile.mean_theta[name]), reverse=True)[:2]
        bound_pressure = {
            name: {
                "upper": profile.fraction_at_upper_bound[name],
                "lower": profile.fraction_at_lower_bound[name],
            }
            for name in dominant_knobs
        }
        print(
            f"{result.name} = {{'dominant_knobs': {dominant_knobs}, "
            f"'bound_pressure': {bound_pressure}}}"
        )
    print(
        "interpretation = This audit determines whether the Pareto sweep exposes genuinely distinct "
        "transition-motor regimes or a compressed, constraint-shaped frontier dominated by one objective "
        "component. It is synthetic ensemble analysis, not experimental validation or full QEC."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transition Grid Atlas research instrument")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to YAML configuration file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("single", help="Run one validated simulation")

    mc_parser = subparsers.add_parser("montecarlo", help="Run Monte Carlo ensemble")
    mc_parser.add_argument("--runs", type=int, default=None, help="Override the number of Monte Carlo runs")

    subparsers.add_parser("sweep", help="Run deterministic parameter sweep")
    subparsers.add_parser("hunter", help="Run recursive search for diffusive candidates")
    subparsers.add_parser("audit", help="Audit ledger evidence under strict modern thresholds")
    subparsers.add_parser("inverse", help="Run inverse transition symmetry analysis")
    hardware_parser = subparsers.add_parser("hardware-map", help="Map a photonic wafer config into effective KTA W and gamma parameters")
    hardware_parser.add_argument(
        "--config",
        dest="hardware_config",
        default=None,
        help="Optional hardware-specific YAML path; accepted after the subcommand for operator convenience",
    )
    hardware_parser.add_argument(
        "--emit-lab-config",
        action="store_true",
        help="Emit an executable KTA lab override config from the hardware mapping",
    )
    hardware_parser.add_argument(
        "--out-config",
        default=None,
        help="Optional output path for the generated lab override config",
    )
    transition_tune_parser = subparsers.add_parser(
        "transition-tune",
        help="Search fixed-grid transition controls for hardware-native error suppression",
    )
    transition_tune_parser.add_argument(
        "--config",
        dest="transition_tune_config",
        default=None,
        help="Optional transition-tuner YAML path; accepted after the subcommand for operator convenience",
    )
    transition_motor_parser = subparsers.add_parser(
        "transition-motor",
        help="Run the instrumented transition motor control stack",
    )
    transition_motor_parser.add_argument(
        "--config",
        dest="transition_motor_config",
        default=None,
        help="Optional transition-motor YAML path; accepted after the subcommand for operator convenience",
    )
    transition_motor_parser.add_argument(
        "--report-sensitivity",
        action="store_true",
        help="Write a sensitivity atlas CSV for the transition motor operating point",
    )
    transition_motor_parser.add_argument(
        "--objective-mode",
        "--operating-mode",
        dest="transition_motor_operating_mode",
        default=None,
        help="Optional objective mode name from the transition-motor registry",
    )
    transition_motor_audit_parser = subparsers.add_parser(
        "transition-motor-audit",
        help="Audit transition motor bound pressure, ablations, limits and seed stability",
    )
    transition_motor_audit_parser.add_argument(
        "--config",
        dest="transition_motor_audit_config",
        default=None,
        help="Optional transition-motor YAML path; accepted after the subcommand for operator convenience",
    )
    transition_motor_ensemble_parser = subparsers.add_parser(
        "transition-motor-ensemble",
        help="Run a synthetic ensemble validation for the transition motor",
    )
    transition_motor_ensemble_parser.add_argument(
        "--config",
        dest="transition_motor_ensemble_config",
        default=None,
        help="Optional transition-motor ensemble YAML path; accepted after the subcommand for operator convenience",
    )
    transition_motor_ensemble_stats_parser = subparsers.add_parser(
        "transition-motor-ensemble-stats",
        help="Compute statistical audit outputs for a transition-motor ensemble CSV",
    )
    transition_motor_ensemble_stats_parser.add_argument("--csv", required=True, help="Input ensemble CSV path")
    transition_motor_ensemble_stats_parser.add_argument("--out-json", required=True, help="Output statistics JSON path")
    transition_motor_ensemble_stats_parser.add_argument("--out-plot", required=True, help="Output statistics plot path")
    transition_motor_pareto_parser = subparsers.add_parser(
        "transition-motor-pareto",
        help="Run an objective-weight Pareto sweep for the transition motor",
    )
    transition_motor_pareto_parser.add_argument(
        "--config",
        dest="transition_motor_pareto_config",
        default=None,
        help="Optional transition-motor Pareto YAML path; accepted after the subcommand for operator convenience",
    )
    transition_motor_pareto_audit_parser = subparsers.add_parser(
        "transition-motor-pareto-audit",
        help="Run a stress audit over Pareto weight presets for the transition motor",
    )
    transition_motor_pareto_audit_parser.add_argument(
        "--config",
        dest="transition_motor_pareto_audit_config",
        default=None,
        help="Optional transition-motor Pareto-audit YAML path; accepted after the subcommand for operator convenience",
    )
    wafer_ensemble_parser = subparsers.add_parser(
        "wafer-ensemble",
        help="Run a synthetic wafer ensemble study over fabrication disorder and phase noise",
    )
    wafer_ensemble_parser.add_argument(
        "--config",
        dest="wafer_ensemble_config",
        default=None,
        help="Optional wafer-ensemble YAML path; accepted after the subcommand for operator convenience",
    )
    lab_parser = subparsers.add_parser("lab", help="Run the Experimental Quantum & RTT Lab and save a trajectory artifact")
    lab_parser.add_argument("--mode", choices=["qm_free", "standard_qm", "anderson", "lindblad", "rtt"], default=None)
    lab_parser.add_argument("--gamma", type=float, default=None, help="Override lab gamma for this run")
    lab_parser.add_argument("--render", action="store_true", help="Render an animation artifact after the lab run")

    animate_parser = subparsers.add_parser("animate", help="Render a GIF or MP4 from a saved lab trajectory artifact")
    animate_parser.add_argument("--trajectory", required=True, help="Path to a saved .npz trajectory artifact")
    animate_parser.add_argument("--out", required=True, help="Output GIF or MP4 path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(
        getattr(args, "hardware_config", None)
        or getattr(args, "transition_tune_config", None)
        or getattr(args, "transition_motor_config", None)
        or getattr(args, "transition_motor_audit_config", None)
        or getattr(args, "transition_motor_ensemble_config", None)
        or getattr(args, "transition_motor_pareto_config", None)
        or getattr(args, "transition_motor_pareto_audit_config", None)
        or getattr(args, "wafer_ensemble_config", None)
        or args.config
    )
    config = load_config(config_path)
    ensure_output_dirs()

    if args.command == "single":
        return run_single_mode(config)
    if args.command == "montecarlo":
        return run_monte_carlo_mode(config, runs=args.runs)
    if args.command == "sweep":
        return run_sweep_mode(config)
    if args.command == "hunter":
        return run_hunter_mode(config)
    if args.command == "audit":
        return run_audit_mode(config)
    if args.command == "inverse":
        return run_inverse_mode(config)
    if args.command == "hardware-map":
        return run_hardware_map_mode(
            config,
            emit_lab_config=args.emit_lab_config,
            out_config=args.out_config,
        )
    if args.command == "transition-tune":
        return run_transition_tune_mode(config)
    if args.command == "transition-motor":
        return run_transition_motor_mode(
            config,
            report_sensitivity=args.report_sensitivity,
            operating_mode=getattr(args, "transition_motor_operating_mode", None),
        )
    if args.command == "transition-motor-audit":
        return run_transition_motor_audit_mode(config)
    if args.command == "transition-motor-ensemble":
        return run_transition_motor_ensemble_mode(config)
    if args.command == "transition-motor-ensemble-stats":
        return run_transition_motor_ensemble_stats_mode(
            csv_path=args.csv,
            out_json=args.out_json,
            out_plot=args.out_plot,
        )
    if args.command == "transition-motor-pareto":
        return run_transition_motor_pareto_mode(config)
    if args.command == "transition-motor-pareto-audit":
        return run_transition_motor_pareto_audit_mode(config_path)
    if args.command == "wafer-ensemble":
        return run_wafer_ensemble_mode(config)
    if args.command == "lab":
        return run_lab_mode(config, mode=args.mode, gamma=args.gamma, render=args.render)
    if args.command == "animate":
        return run_animate_mode(trajectory=args.trajectory, output_path=args.out)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
