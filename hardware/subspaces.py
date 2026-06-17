"""Subspace helpers for hardware-native error-suppression studies."""

from __future__ import annotations

import numpy as np


def orthonormalize_columns(matrix: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError("matrix must be 2D")

    basis: list[np.ndarray] = []
    for column_index in range(array.shape[1]):
        vector = np.array(array[:, column_index], dtype=np.complex128, copy=True)
        for basis_vector in basis:
            vector -= basis_vector * np.vdot(basis_vector, vector)
        norm = float(np.linalg.norm(vector))
        if norm > tol:
            basis.append(vector / norm)

    if not basis:
        return np.zeros((array.shape[0], 0), dtype=np.complex128)
    return np.column_stack(basis)


def information_subspace(
    n_sites: int,
    input_index: int,
    target_indices: list[int],
) -> np.ndarray:
    """Return a basis for the information-carrying subspace."""

    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if input_index < 0 or input_index >= n_sites:
        raise ValueError("input_index out of range")
    if not target_indices:
        raise ValueError("target_indices must be non-empty")

    target_set = sorted({int(index) for index in target_indices})
    for index in target_set:
        if index < 0 or index >= n_sites:
            raise ValueError(f"target index out of range: {index}")

    input_mode = np.zeros(n_sites, dtype=np.complex128)
    input_mode[input_index] = 1.0

    target_mode = np.zeros(n_sites, dtype=np.complex128)
    amplitude = 1.0 / np.sqrt(float(len(target_set)))
    for index in target_set:
        target_mode[index] = amplitude

    return orthonormalize_columns(np.column_stack([input_mode, target_mode]))


def transport_subspace_from_hamiltonian(
    H: np.ndarray,
    input_index: int,
    target_indices: list[int],
    *,
    n_modes: int = 2,
) -> np.ndarray:
    """Return an H-dependent transport subspace.

    The selected modes are the Hamiltonian eigenmodes with the largest combined
    overlap with the input mode and target detector sector. This is distinct
    from the fixed geometric input/target subspace.
    """

    operator = np.asarray(H, dtype=np.complex128)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("H must be a square matrix")
    n_sites = operator.shape[0]
    if input_index < 0 or input_index >= n_sites:
        raise ValueError("input_index out of range")
    if not target_indices:
        raise ValueError("target_indices must be non-empty")
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")

    target_set = sorted({int(index) for index in target_indices})
    for index in target_set:
        if index < 0 or index >= n_sites:
            raise ValueError(f"target index out of range: {index}")

    eigvals, eigvecs = np.linalg.eigh(operator)
    del eigvals  # eigenvalues are not needed for the selection itself

    scores = []
    for mode_index in range(eigvecs.shape[1]):
        vector = eigvecs[:, mode_index]
        score = float(abs(vector[input_index]) ** 2 + np.sum(np.abs(vector[target_set]) ** 2))
        scores.append((score, mode_index))

    selected = [index for _, index in sorted(scores, reverse=True)[: min(n_modes, eigvecs.shape[1])]]
    return orthonormalize_columns(eigvecs[:, selected])


def projector_from_basis(U: np.ndarray) -> np.ndarray:
    basis = np.asarray(U, dtype=np.complex128)
    if basis.ndim != 2:
        raise ValueError("U must be a 2D matrix")
    return basis @ np.conjugate(basis.T)


def complement_projector(P: np.ndarray) -> np.ndarray:
    projector = np.asarray(P, dtype=np.complex128)
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise ValueError("P must be a square matrix")
    identity = np.eye(projector.shape[0], dtype=np.complex128)
    return identity - projector


def diagonal_phase_noise(profile: np.ndarray) -> np.ndarray:
    vector = np.asarray(profile, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("profile must be a 1D array")
    return np.diag(vector.astype(np.complex128, copy=False))


def noise_overlap_with_subspace(
    noise_ops: list[np.ndarray],
    U_info: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    basis = np.asarray(U_info, dtype=np.complex128)
    if basis.ndim != 2:
        raise ValueError("U_info must be a 2D matrix")

    numerator = 0.0
    denominator = 0.0
    basis_dagger = np.conjugate(basis.T)
    for noise_operator in noise_ops:
        operator = np.asarray(noise_operator, dtype=np.complex128)
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError("noise operators must be square matrices")
        projected = basis_dagger @ operator @ basis
        numerator += float(np.linalg.norm(projected, ord="fro") ** 2)
        denominator += float(np.linalg.norm(operator, ord="fro") ** 2)

    overlap = numerator / (denominator + float(eps))
    return float(np.clip(overlap, 0.0, 1.0))


def suppression_score(
    noise_ops: list[np.ndarray],
    U_info: np.ndarray,
) -> float:
    score = 1.0 - noise_overlap_with_subspace(noise_ops, U_info)
    return float(np.clip(score, 0.0, 1.0))
