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
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from engine import run_transport_simulation
from explorer.parameter_sweep import detect_phase_boundary_zones, run_parameter_sweep
from explorer.phase_map import build_phase_matrix, save_phase_map_plot
from explorer.recursive_hunter import run_recursive_hunter
from validation.audit import build_audit_report
from validation.monte_carlo import summarise_samples


PROJECT_ROOT = Path(__file__).resolve().parent
ATLAS_RESULTS_DIR = PROJECT_ROOT / "atlas" / "results"
ATLAS_PLOTS_DIR = PROJECT_ROOT / "atlas" / "plots"
ATLAS_REPORTS_DIR = PROJECT_ROOT / "atlas" / "reports"
MASTER_RESULTS_PATH = ATLAS_RESULTS_DIR / "master_results.csv"
LATEST_REPORT_PATH = ATLAS_REPORTS_DIR / "latest_report.md"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs() -> None:
    ATLAS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ATLAS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ATLAS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def timestamp_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))
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
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
