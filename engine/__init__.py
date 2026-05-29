"""Core simulation orchestration for Transition Grid Atlas.

This package deliberately keeps the quantum/statistical simulation engine
separate from exploration and reporting layers. The helper function
``run_transport_simulation`` provides a deterministic, validated simulation
entry point that other modules can call without knowing anything about plots,
CSV files, or markdown reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.evolution import simulate_dynamics
from engine.hamiltonian import build_initial_state, build_tight_binding_hamiltonian
from engine.observables import classify_transport_regime, compute_transport_metrics
from validation.hermitian import hermitian_error
from validation.unitarity import unitarity_error


def run_transport_simulation(
    config: dict[str, Any],
    parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run one deterministic transport simulation and return diagnostics.

    Parameters are merged over the base values from ``config["parameters"]``.
    All real-valued physical arrays are created as ``float64`` and all quantum
    state arrays as ``complex128`` to preserve numerical stability.
    """

    sim_cfg = dict(config["simulation"])
    base_parameters = dict(config["parameters"])
    if parameters:
        base_parameters.update(parameters)

    size = int(sim_cfg["grid_size"])
    hopping = float(sim_cfg["hopping"])
    dt = float(sim_cfg["dt"])
    steps = int(sim_cfg.get("t_max_steps", sim_cfg["steps"]))
    fit_start_step = int(sim_cfg.get("fit_start_step", 1))
    fit_end_step = sim_cfg.get("fit_end_step")
    tolerance = float(sim_cfg.get("validation_tolerance", 1.0e-12))

    parameter_block = {
        "W": float(base_parameters["W"]),
        "bias": float(base_parameters["bias"]),
        "eta": float(base_parameters["eta"]),
        "gamma": float(base_parameters["gamma"]),
        "seed": int(base_parameters["seed"]),
    }

    hamiltonian = build_tight_binding_hamiltonian(
        size=size,
        hopping=hopping,
        disorder_strength=parameter_block["W"],
        bias=parameter_block["bias"],
        seed=parameter_block["seed"],
    )
    initial_state = build_initial_state(size=size, initial_site=sim_cfg.get("initial_site", "center"))
    evolution = simulate_dynamics(
        base_hamiltonian=hamiltonian,
        initial_state=initial_state,
        dt=dt,
        steps=steps,
        eta=parameter_block["eta"],
        gamma=parameter_block["gamma"],
        feedback_smoothing=float(sim_cfg.get("feedback_smoothing", 1.0)),
    )
    metrics = compute_transport_metrics(
        trajectory=evolution["trajectory"],
        dt=dt,
        hopping=hopping,
        fit_start_step=fit_start_step,
        fit_end_step=fit_end_step,
    )

    hermitian_errors = [hermitian_error(step_h) for step_h in evolution["step_hamiltonians"]]
    max_hermitian_error = float(max(hermitian_errors)) if hermitian_errors else hermitian_error(hamiltonian)
    total_unitarity_error = unitarity_error(evolution["total_operator"])
    status = classify_transport_regime(
        alpha=metrics["alpha"],
        r2=metrics["r2"],
        unitarity_error_value=total_unitarity_error,
        hermitian_error_value=max_hermitian_error,
        scaling_window_stable=metrics["scaling_window_stable"],
        tolerance=tolerance,
    )

    result = {
        "parameters": parameter_block,
        "alpha": metrics["alpha"],
        "early_alpha": metrics["early_alpha"],
        "late_alpha": metrics["late_alpha"],
        "alpha_window_shift": metrics["alpha_window_shift"],
        "scaling_window_stable": metrics["scaling_window_stable"],
        "r2": metrics["r2"],
        "early_r2": metrics["early_r2"],
        "late_r2": metrics["late_r2"],
        "regression_rmse": metrics["regression_rmse"],
        "mean_current": metrics["mean_current"],
        "unitarity_error": total_unitarity_error,
        "hermitian_error": max_hermitian_error,
        "status": status,
        "times": metrics["times"],
        "msd": metrics["msd"],
        "sigma": metrics["sigma"],
        "current_series": metrics["current_series"],
        "trajectory": evolution["trajectory"],
        "total_operator": evolution["total_operator"],
        "hamiltonian": hamiltonian,
        "step_hamiltonians": evolution["step_hamiltonians"],
    }
    return result
