"""Resolve patient age for vital sign thresholds (intake + optional overrides).

Re-exports shared helpers from ``ed_triage.common.age`` for import stability.
"""

from __future__ import annotations

from ed_triage.common.age import AgeTriple, age_known, resolve_age_for_vitals

__all__ = ["AgeTriple", "age_known", "resolve_age_for_vitals"]
