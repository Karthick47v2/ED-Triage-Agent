"""Map deterministic vital assessments to CRA Phase 2 RAG query boosts."""

from __future__ import annotations

from ed_triage.common.vital_assessment.tables import (
    VitalSignsAssessment,
    VitalStatus,
)

_BOOST_STATUSES = frozenset({
    VitalStatus.CRITICAL,
    VitalStatus.HIGH_RISK,
})

# Keywords keyed by VitalAssessment.name from assess.py.
_RAG_KEYWORDS_BY_VITAL: dict[str, str] = {
    "Heart Rate": "tachycardia bradycardia",
    "Respiratory Rate": "respiratory distress",
    "Systolic Blood Pressure": "hypotension shock",
    "Diastolic Blood Pressure": "hypertensive urgency",
    "Oxygen Saturation": "hypoxia respiratory distress",
    "Temperature": "fever hypothermia",
}


def vital_rag_query_boost(assessment: VitalSignsAssessment) -> str:
    """Return extra RAG query terms for CRITICAL / HIGH_RISK vitals only."""
    terms: list[str] = []
    seen: set[str] = set()
    for vital in assessment.vitals:
        if vital.status not in _BOOST_STATUSES:
            continue
        keywords = _RAG_KEYWORDS_BY_VITAL.get(vital.name)
        if keywords and keywords not in seen:
            seen.add(keywords)
            terms.append(keywords)
    return f" {' '.join(terms)}" if terms else ""
