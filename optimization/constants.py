"""Domain constants for the optimization module."""

from __future__ import annotations

INVALID_MOLECULE_OBJECTIVE: float = -1.0
"""Objective value assigned to candidates that cannot be decoded or scored.

Since pyribs maximises objectives, a value below ``threshold_min`` ensures
invalid molecules are never inserted into the archive.
"""
