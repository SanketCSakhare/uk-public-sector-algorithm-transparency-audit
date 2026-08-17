"""Transparent, non-normative disclosure indicators for ATRS records."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .parser import normalise_space


NON_ANSWERS = {
    "",
    "-",
    "n/a",
    "na",
    "none",
    "not applicable",
    "not available",
    "unknown",
    "tbc",
    "tbd",
}


@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    patterns: tuple[str, ...]
    description: str


INDICATORS = (
    Indicator(
        "accountability",
        "Named accountability",
        (r"senior responsible owner", r"contact email"),
        "A substantive owner or contact disclosure is present.",
    ),
    Indicator(
        "human_oversight",
        "Human oversight",
        (r"human review", r"human.*oversight", r"human.*decision"),
        "The record describes human review or decision responsibility.",
    ),
    Indicator(
        "contestability",
        "Appeal or review route",
        (r"appeals? and review", r"appeal", r"contest"),
        "The record addresses an appeal, review, or contestability route.",
    ),
    Indicator(
        "impact_assessment",
        "Impact assessment",
        (r"impact assessments?", r"data protection impact", r"equality impact"),
        "The record addresses impact assessment activity.",
    ),
    Indicator(
        "risks_mitigations",
        "Risks and mitigations",
        (r"risks? and mitigations?", r"risk.*mitigation"),
        "The record provides substantive risk or mitigation information.",
    ),
    Indicator(
        "model_performance",
        "Performance evidence",
        (r"model performance", r"performance metrics?", r"accuracy"),
        "The record provides substantive model or tool performance information.",
    ),
    Indicator(
        "data_governance",
        "Operational data governance",
        (r"data sources?", r"data access and storage", r"data sharing agreements?"),
        "The record describes operational data sources, access, storage, or sharing.",
    ),
    Indicator(
        "monitoring_maintenance",
        "Monitoring or maintenance",
        (r"maintenance", r"monitoring", r"ongoing review"),
        "The record describes maintenance, monitoring, or ongoing review.",
    ),
)


def is_substantive(value: str, minimum_chars: int = 12) -> bool:
    cleaned = normalise_space(value).casefold().strip(" .;:")
    if cleaned in NON_ANSWERS:
        return False
    if any(cleaned.startswith(prefix) for prefix in ("n/a -", "n/a:", "not applicable -")):
        # A reason after a non-applicable marker is still a disclosure.
        return len(cleaned) >= minimum_chars + 6
    return len(cleaned) >= minimum_chars


def score_sections(sections: dict[str, dict[str, str]]) -> dict[str, object]:
    searchable = [
        (f"{key} {item.get('heading', '')}".casefold(), item.get("value", ""))
        for key, item in sections.items()
    ]
    results: dict[str, bool] = {}
    evidence: dict[str, list[str]] = {}
    for indicator in INDICATORS:
        matches: list[str] = []
        for heading, value in searchable:
            if any(re.search(pattern, heading) for pattern in indicator.patterns):
                if is_substantive(value):
                    matches.append(normalise_space(value)[:180])
        results[indicator.key] = bool(matches)
        evidence[indicator.key] = matches[:2]

    substantive_fields = sum(
        1 for item in sections.values() if is_substantive(item.get("value", ""))
    )
    indicator_count = sum(results.values())
    return {
        **results,
        "indicator_count": indicator_count,
        "disclosure_score": round(100 * indicator_count / len(INDICATORS), 1),
        "field_count": len(sections),
        "substantive_field_count": substantive_fields,
        "field_completeness_ratio": round(
            substantive_fields / len(sections), 3
        ) if sections else 0.0,
        "evidence": evidence,
    }
