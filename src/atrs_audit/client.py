"""Small standard-library client for official GOV.UK APIs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


GOVUK_ROOT = "https://www.gov.uk"
SEARCH_ENDPOINT = f"{GOVUK_ROOT}/api/search.json"
SEARCH_FIELDS = (
    "title",
    "description",
    "link",
    "public_timestamp",
    "algorithmic_transparency_record_organisation",
    "algorithmic_transparency_record_organisation_type",
    "algorithmic_transparency_record_function",
    "algorithmic_transparency_record_capability",
    "algorithmic_transparency_record_phase",
    "algorithmic_transparency_record_region",
    "algorithmic_transparency_record_date_published",
    "algorithmic_transparency_record_atrs_version",
)


@dataclass
class FetchFailure:
    link: str
    error: str


class GovUKClient:
    def __init__(self, timeout: int = 30, retries: int = 3) -> None:
        self.timeout = timeout
        self.retries = retries
        self.user_agent = (
            "uk-atrs-governance-audit/1.0 "
            "(public-interest research; source: github.com/SanketCSakhare)"
        )

    def get_json(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Failed to fetch {url}: {last_error}")

    def fetch_catalog(self, count: int = 500) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {
                "filter_document_type": "algorithmic_transparency_record",
                "count": count,
                "fields": ",".join(SEARCH_FIELDS),
                "order": "-public_timestamp",
            }
        )
        payload = self.get_json(f"{SEARCH_ENDPOINT}?{query}")
        if payload.get("total", 0) > len(payload.get("results", [])):
            raise RuntimeError(
                "GOV.UK returned fewer records than the reported total; increase count."
            )
        return payload

    def fetch_record(self, link: str) -> dict[str, Any]:
        if not link.startswith("/algorithmic-transparency-records/"):
            raise ValueError(f"Unexpected ATRS link: {link}")
        return self.get_json(f"{GOVUK_ROOT}/api/content{link}")

    def fetch_records(
        self, catalog: dict[str, Any], max_workers: int = 8
    ) -> tuple[list[dict[str, Any]], list[FetchFailure]]:
        items = catalog.get("results", [])
        records: list[dict[str, Any]] = []
        failures: list[FetchFailure] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.fetch_record, item["link"]): item for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    detail = future.result()
                    records.append({"catalog": item, "detail": detail})
                except Exception as exc:  # Keep a complete, inspectable failure log.
                    failures.append(FetchFailure(item.get("link", ""), str(exc)))
        records.sort(key=lambda item: item["catalog"].get("link", ""))
        return records, failures
