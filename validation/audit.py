"""Audit helpers for Transition Grid Atlas research ledgers.

This module is intentionally standard-library only. Its job is not to simulate
or validate physics directly, but to inspect already written ledger CSV files
and report which claims are still supported under the current strict criteria.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_CONFIG = {
    "diffusive_alpha_target": 0.5,
    "diffusive_alpha_tolerance": 0.035,
    "min_r2": 0.95,
    "max_unitarity_error": 1.0e-12,
    "max_hermitian_error": 1.0e-12,
    "required_gamma": 0.0,
    "max_alpha_window_shift": 0.05,
    "top_n": 20,
}


def parse_float(value: Any) -> float | None:
    """Parse a float-like value or return None for blank/invalid input."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_bool(value: Any) -> bool | None:
    """Parse common textual booleans while keeping blank legacy fields as None."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows or return an empty list when the file is absent."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count status values in a ledger-like CSV row list."""

    counts = Counter(str(row.get("status", "")).strip() for row in rows if str(row.get("status", "")).strip())
    return dict(sorted(counts.items()))


def merge_audit_config(config: dict[str, Any]) -> dict[str, float | int]:
    merged = dict(DEFAULT_AUDIT_CONFIG)
    merged.update(config.get("audit", {}))
    return merged


def is_modern_strict_stable_diffusive(
    row: dict[str, str],
    thresholds: dict[str, float | int],
) -> bool:
    """Return True only for rows that survive the modern strict audit gate."""

    alpha = parse_float(row.get("alpha"))
    r2 = parse_float(row.get("r2"))
    unitarity_error = parse_float(row.get("unitarity_error"))
    hermitian_error = parse_float(row.get("hermitian_error"))
    gamma = parse_float(row.get("gamma"))
    alpha_window_shift = parse_float(row.get("alpha_window_shift"))
    scaling_window_stable = parse_bool(row.get("scaling_window_stable"))

    if None in (alpha, r2, unitarity_error, hermitian_error, gamma, alpha_window_shift):
        return False
    if scaling_window_stable is not True:
        return False

    return (
        abs(alpha - float(thresholds["diffusive_alpha_target"])) <= float(thresholds["diffusive_alpha_tolerance"])
        and r2 >= float(thresholds["min_r2"])
        and unitarity_error < float(thresholds["max_unitarity_error"])
        and hermitian_error < float(thresholds["max_hermitian_error"])
        and abs(gamma - float(thresholds["required_gamma"])) <= 0.0
        and alpha_window_shift <= float(thresholds["max_alpha_window_shift"])
    )


def is_transient_diffusive_like(
    row: dict[str, str],
    thresholds: dict[str, float | int],
) -> bool:
    """Identify near-diffusive rows rejected specifically by transient logic."""

    alpha = parse_float(row.get("alpha"))
    r2 = parse_float(row.get("r2"))
    unitarity_error = parse_float(row.get("unitarity_error"))
    hermitian_error = parse_float(row.get("hermitian_error"))
    gamma = parse_float(row.get("gamma"))
    alpha_window_shift = parse_float(row.get("alpha_window_shift"))
    scaling_window_stable = parse_bool(row.get("scaling_window_stable"))

    if None in (alpha, r2, unitarity_error, hermitian_error, gamma):
        return False

    base_diffusive_like = (
        abs(alpha - float(thresholds["diffusive_alpha_target"])) <= float(thresholds["diffusive_alpha_tolerance"])
        and r2 >= float(thresholds["min_r2"])
        and unitarity_error < float(thresholds["max_unitarity_error"])
        and hermitian_error < float(thresholds["max_hermitian_error"])
        and abs(gamma - float(thresholds["required_gamma"])) <= 0.0
    )
    if not base_diffusive_like:
        return False

    if scaling_window_stable is False:
        return True
    if scaling_window_stable is True and alpha_window_shift is not None:
        return alpha_window_shift > float(thresholds["max_alpha_window_shift"])
    return False


def duplicate_parameter_keys(rows: list[dict[str, str]]) -> list[tuple[tuple[str, str, str, str, str], int]]:
    """Count duplicate parameter signatures using (W, eta, gamma, seed, mode)."""

    keys = Counter(
        (
            str(row.get("W", "")).strip(),
            str(row.get("eta", "")).strip(),
            str(row.get("gamma", "")).strip(),
            str(row.get("seed", "")).strip(),
            str(row.get("mode", "")).strip(),
        )
        for row in rows
    )
    duplicates = [(key, count) for key, count in keys.items() if count > 1]
    duplicates.sort(key=lambda item: (-item[1], item[0]))
    return duplicates


def find_latest_sweep_file(results_dir: Path) -> Path | None:
    """Find the lexicographically latest sweep CSV file."""

    sweep_files = sorted(results_dir.glob("sweep_*.csv"))
    return sweep_files[-1] if sweep_files else None


def _transient_sort_key(row: dict[str, str], thresholds: dict[str, float | int]) -> tuple[float, float, float]:
    alpha = parse_float(row.get("alpha"))
    r2 = parse_float(row.get("r2"))
    unitarity_error = parse_float(row.get("unitarity_error"))
    return (
        abs((alpha or 0.0) - float(thresholds["diffusive_alpha_target"])),
        -(r2 or 0.0),
        unitarity_error if unitarity_error is not None else float("inf"),
    )


def _row_source_label(row: dict[str, str], default_source: str) -> str:
    if row.get("source_batch"):
        return str(row["source_batch"])
    return default_source


def _parameter_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("W", "")).strip(),
        str(row.get("eta", "")).strip(),
        str(row.get("gamma", "")).strip(),
        str(row.get("seed", "")).strip(),
        str(row.get("mode", "")).strip(),
    )


def _transient_parameter_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("W", "")).strip(),
        str(row.get("eta", "")).strip(),
        str(row.get("gamma", "")).strip(),
        str(row.get("seed", "")).strip(),
    )


def _transient_dedup_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    alpha_window_shift = parse_float(row.get("alpha_window_shift"))
    r2 = parse_float(row.get("r2"))
    unitarity_error = parse_float(row.get("unitarity_error"))
    alpha = parse_float(row.get("alpha"))
    return (
        alpha_window_shift if alpha_window_shift is not None else float("inf"),
        -(r2 or 0.0),
        unitarity_error if unitarity_error is not None else float("inf"),
        abs(alpha - 0.5) if alpha is not None else float("inf"),
    )


def deduplicate_transient_rows(
    rows: list[dict[str, str]],
    thresholds: dict[str, float | int],
) -> list[dict[str, str]]:
    """Keep only the best transient evidence row per parameter key."""

    best_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = _transient_parameter_key(row)
        current = best_by_key.get(key)
        if current is None or _transient_dedup_key(row) < _transient_dedup_key(current):
            best_by_key[key] = row
    deduplicated = list(best_by_key.values())
    deduplicated.sort(key=lambda row: _transient_sort_key(row, thresholds))
    return deduplicated


def build_audit_report(
    config: dict[str, Any],
    project_root: Path,
) -> str:
    """Build a conservative text audit report from atlas ledger files."""

    thresholds = merge_audit_config(config)
    results_dir = project_root / "atlas" / "results"
    master_path = results_dir / "master_results.csv"
    latest_sweep_path = find_latest_sweep_file(results_dir)

    master_rows = load_csv_rows(master_path)
    latest_sweep_rows = load_csv_rows(latest_sweep_path) if latest_sweep_path else []

    master_counts = status_counts(master_rows)
    latest_sweep_counts = status_counts(latest_sweep_rows)
    legacy_diffusive_count = sum(str(row.get("status", "")).strip() == "VALID_DIFFUSIVE_CANDIDATE" for row in master_rows)
    modern_strict_rows = [row for row in master_rows if is_modern_strict_stable_diffusive(row, thresholds)]
    master_transient_rows = deduplicate_transient_rows(
        [row for row in master_rows if is_transient_diffusive_like(row, thresholds)],
        thresholds,
    )
    latest_sweep_transient_rows = deduplicate_transient_rows(
        [row for row in latest_sweep_rows if is_transient_diffusive_like(row, thresholds)],
        thresholds,
    )
    duplicates = duplicate_parameter_keys(master_rows)

    top_n = int(thresholds["top_n"])
    latest_sweep_name = latest_sweep_path.name if latest_sweep_path else None

    lines = [
        "Transition Grid Atlas Audit",
        "============================",
        f"master_results.csv: {master_path}",
        f"latest sweep file: {latest_sweep_name or 'missing'}",
        "",
        f"1. Total master ledger rows: {len(master_rows)}",
        "2. Master status counts:",
    ]
    if master_counts:
        lines.extend([f"   - {status}: {count}" for status, count in master_counts.items()])
    else:
        lines.append("   - no master ledger rows found")

    lines.extend(
        [
            f"3. Latest sweep filename: {latest_sweep_name or 'missing'}",
            f"4. Latest sweep row count: {len(latest_sweep_rows)}",
            "5. Latest sweep status counts:",
        ]
    )
    if latest_sweep_counts:
        lines.extend([f"   - {status}: {count}" for status, count in latest_sweep_counts.items()])
    else:
        lines.append("   - no latest sweep rows found")

    lines.extend(
        [
            f"6. Legacy VALID_DIFFUSIVE_CANDIDATE rows in master_results.csv: {legacy_diffusive_count}",
            f"7. Modern strict stable diffusive candidates: {len(modern_strict_rows)}",
            "",
            "8A. Top transient diffusive-like rows in master ledger:",
        ]
    )
    if master_transient_rows:
        for row in master_transient_rows[:top_n]:
            lines.append(
                "   - "
                f"status={row.get('status','')}, "
                f"W={row.get('W','')}, eta={row.get('eta','')}, gamma={row.get('gamma','')}, "
                f"seed={row.get('seed','')}, alpha={row.get('alpha','')}, r2={row.get('r2','')}, "
                f"unitarity_error={row.get('unitarity_error','')}, "
                f"alpha_window_shift={row.get('alpha_window_shift','')}, "
                f"scaling_window_stable={row.get('scaling_window_stable','')}, "
                f"source={_row_source_label(row, 'master_results.csv')}"
            )
    else:
        lines.append("   - no transient diffusive-like master-ledger rows matched the configured filters")

    lines.extend(["", "8B. Top transient diffusive-like rows in latest sweep:"])
    if latest_sweep_transient_rows:
        for row in latest_sweep_transient_rows[:top_n]:
            lines.append(
                "   - "
                f"status={row.get('status','')}, "
                f"W={row.get('W','')}, eta={row.get('eta','')}, gamma={row.get('gamma','')}, "
                f"seed={row.get('seed','')}, alpha={row.get('alpha','')}, r2={row.get('r2','')}, "
                f"unitarity_error={row.get('unitarity_error','')}, "
                f"alpha_window_shift={row.get('alpha_window_shift','')}, "
                f"scaling_window_stable={row.get('scaling_window_stable','')}, "
                f"source={_row_source_label(row, latest_sweep_name or 'latest_sweep.csv')}"
            )
    else:
        lines.append("   - no transient diffusive-like latest-sweep rows matched the configured filters")

    lines.extend(
        [
            "",
            "9. Duplicate parameter keys in master_results.csv:",
        ]
    )
    if duplicates:
        for (W, eta, gamma, seed, mode), count in duplicates[:top_n]:
            lines.append(
                f"   - key=(W={W}, eta={eta}, gamma={gamma}, seed={seed}, mode={mode}) count={count}"
            )
    else:
        lines.append("   - no duplicate parameter keys detected")

    lines.extend(["", "10. Scientific interpretation:"])
    if len(modern_strict_rows) == 0:
        lines.append(
            "No modern strict stable diffusive candidate is currently supported. "
            "Diffusive-like alpha values exist, but current two-window scaling marks them as transient/weak."
        )
    else:
        lines.append("Modern strict stable diffusive candidates exist under the configured thresholds.")

    return "\n".join(lines)
