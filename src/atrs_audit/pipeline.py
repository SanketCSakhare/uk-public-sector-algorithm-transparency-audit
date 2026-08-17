"""Transform GOV.UK ATRS records into an auditable flat dataset."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .client import GOVUK_ROOT, GovUKClient
from .parser import parse_sections
from .report import build_report
from .scoring import INDICATORS, score_sections


def as_pipe_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def normalise_atrs_version(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    raw = raw.removeprefix("version").removeprefix("v").strip()
    if raw in {"1.1", "2.1", "3", "3.0", "4", "4.0"}:
        major_minor = f"{raw}.0" if raw in {"3", "4"} else raw
        return f"v{major_minor}"
    return str(value or "").strip()


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    catalog = record["catalog"]
    detail = record["detail"]
    metadata = detail.get("details", {}).get("metadata", {})
    sections = parse_sections(detail.get("details", {}).get("body", ""))
    score = score_sections(sections)
    raw_version = metadata.get(
        "algorithmic_transparency_record_atrs_version",
        catalog.get("algorithmic_transparency_record_atrs_version", ""),
    )
    return {
        "record_id": detail.get("content_id", ""),
        "title": detail.get("title") or catalog.get("title", ""),
        "description": detail.get("description") or catalog.get("description", ""),
        "source_url": f"{GOVUK_ROOT}{catalog.get('link', '')}",
        "organisation": metadata.get(
            "algorithmic_transparency_record_organisation",
            catalog.get("algorithmic_transparency_record_organisation", ""),
        ),
        "organisation_type": as_pipe_list(metadata.get(
            "algorithmic_transparency_record_organisation_type",
            catalog.get("algorithmic_transparency_record_organisation_type", []),
        )),
        "function": as_pipe_list(metadata.get(
            "algorithmic_transparency_record_function",
            catalog.get("algorithmic_transparency_record_function", []),
        )),
        "capability": as_pipe_list(metadata.get(
            "algorithmic_transparency_record_capability",
            catalog.get("algorithmic_transparency_record_capability", []),
        )),
        "phase": metadata.get(
            "algorithmic_transparency_record_phase",
            catalog.get("algorithmic_transparency_record_phase", ""),
        ),
        "region": as_pipe_list(metadata.get(
            "algorithmic_transparency_record_region",
            catalog.get("algorithmic_transparency_record_region", []),
        )),
        "date_published": metadata.get(
            "algorithmic_transparency_record_date_published",
            catalog.get("algorithmic_transparency_record_date_published", ""),
        ),
        "atrs_version_raw": raw_version,
        "atrs_version": normalise_atrs_version(raw_version),
        "public_updated_at": detail.get("public_updated_at", ""),
        **{indicator.key: score[indicator.key] for indicator in INDICATORS},
        "indicator_count": score["indicator_count"],
        "disclosure_score": score["disclosure_score"],
        "field_count": score["field_count"],
        "substantive_field_count": score["substantive_field_count"],
        "field_completeness_ratio": score["field_completeness_ratio"],
    }


def quality_checks(rows: list[dict[str, Any]], reported_total: int) -> dict[str, Any]:
    urls = [row["source_url"] for row in rows]
    valid_phases = {
        "pre-deployment", "beta-pilot", "private-beta", "public-beta",
        "production", "retired", "",
    }
    return {
        "reported_total": reported_total,
        "records_fetched": len(rows),
        "record_count_matches_source": len(rows) == reported_total,
        "duplicate_source_urls": len(urls) - len(set(urls)),
        "missing_title": sum(not row["title"] for row in rows),
        "missing_organisation": sum(not row["organisation"] for row in rows),
        "missing_publication_date": sum(not row["date_published"] for row in rows),
        "missing_atrs_version": sum(not row["atrs_version_raw"] for row in rows),
        "unknown_phase_values": sorted(
            {str(row["phase"]) for row in rows if row["phase"] not in valid_phases}
        ),
        "raw_version_labels": dict(
            sorted(Counter(row["atrs_version_raw"] for row in rows).items())
        ),
        "normalised_versions": dict(
            sorted(Counter(row["atrs_version"] for row in rows).items())
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty audit dataset")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run(output_root: Path, max_workers: int = 8) -> dict[str, Any]:
    client = GovUKClient()
    catalog = client.fetch_catalog()
    records, failures = client.fetch_records(catalog, max_workers=max_workers)
    if failures:
        failed = ", ".join(failure.link for failure in failures[:5])
        raise RuntimeError(f"Could not fetch {len(failures)} records: {failed}")

    rows = [flatten_record(record) for record in records]
    checks = quality_checks(rows, int(catalog.get("total", 0)))
    if not checks["record_count_matches_source"] or checks["duplicate_source_urls"]:
        raise RuntimeError(f"Blocking data-quality failure: {checks}")

    generated_at = datetime.now(timezone.utc).isoformat()
    source = {
        "source_name": "GOV.UK Algorithmic Transparency Records",
        "source_url": f"{GOVUK_ROOT}/algorithmic-transparency-records",
        "api_url": f"{GOVUK_ROOT}/api/search.json",
        "retrieved_at_utc": generated_at,
        "licence": "Open Government Licence v3.0",
        "reported_total": catalog.get("total", 0),
    }
    snapshot = {
        "source": source,
        "results": catalog.get("results", []),
    }
    write_json(output_root / "data/snapshot/atrs_metadata.json", snapshot)
    write_csv(output_root / "data/processed/atrs_audit.csv", rows)
    summary = build_report(rows, checks, source, output_root / "reports/index.html")
    write_json(output_root / "reports/summary.json", summary)
    write_json(output_root / "reports/data_quality.json", checks)
    return summary
