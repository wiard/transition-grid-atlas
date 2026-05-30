"""Backward-compatible wrapper around KTA audit helpers."""

from __future__ import annotations

from validation.kta_audit import late_window_metrics, summarize_kta_audit as summarize_lab_audit

__all__ = ["late_window_metrics", "summarize_lab_audit"]
