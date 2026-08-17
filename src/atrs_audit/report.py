"""Generate a portable, dependency-free HTML governance audit."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .scoring import INDICATORS


def split_values(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    values: Counter[str] = Counter()
    for row in rows:
        for value in str(row.get(field, "")).split("|"):
            if value:
                values[value] += 1
    return values


def label(value: str) -> str:
    return value.replace("-", " ").title()


def bar_rows(items: Iterable[tuple[str, int]], total: int) -> str:
    rendered = []
    for name, count in items:
        pct = 100 * count / total if total else 0
        rendered.append(
            f'<div class="bar-row"><span>{html.escape(label(name))}</span>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div></div>'
            f'<strong>{count}</strong><small>{pct:.1f}%</small></div>'
        )
    return "".join(rendered)


def build_report(
    rows: list[dict[str, Any]],
    checks: dict[str, Any],
    source: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    total = len(rows)
    organisations = split_values(rows, "organisation")
    phases = split_values(rows, "phase")
    functions = split_values(rows, "function")
    versions = split_values(rows, "atrs_version")
    indicator_rates = {
        indicator.key: round(100 * sum(bool(row[indicator.key]) for row in rows) / total, 1)
        for indicator in INDICATORS
    }
    latest = max((row["date_published"] for row in rows if row["date_published"]), default="")
    summary = {
        "record_count": total,
        "organisation_count": len(organisations),
        "latest_publication_date": latest,
        "mean_disclosure_score": round(mean(float(row["disclosure_score"]) for row in rows), 1),
        "indicator_coverage_percent": indicator_rates,
        "phase_counts": dict(phases.most_common()),
        "function_counts": dict(functions.most_common()),
        "atrs_version_counts": dict(versions.most_common()),
        "data_quality": checks,
        "source": source,
    }
    coverage_rows = "".join(
        f"<tr><td>{html.escape(indicator.label)}</td>"
        f"<td>{indicator_rates[indicator.key]:.1f}%</td>"
        f"<td>{sum(bool(row[indicator.key]) for row in rows)} of {total}</td>"
        f"<td>{html.escape(indicator.description)}</td></tr>"
        for indicator in INDICATORS
    )
    top_orgs = bar_rows(organisations.most_common(10), total)
    phase_bars = bar_rows(phases.most_common(), total)
    function_bars = bar_rows(functions.most_common(), total)
    qa_status = "PASS" if (
        checks["record_count_matches_source"]
        and checks["duplicate_source_urls"] == 0
        and checks["missing_title"] == 0
    ) else "REVIEW"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UK Public Sector Algorithm Transparency Audit</title>
<style>
:root{{--ink:#17202a;--muted:#5d6d7e;--paper:#fbfaf7;--card:#fff;--navy:#12304a;--teal:#007f78;--gold:#d9a441;--line:#dfe5e8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
header{{background:linear-gradient(130deg,var(--navy),#1c5870);color:white;padding:64px 24px 52px}} .wrap{{max-width:1100px;margin:auto}}
h1{{font-size:clamp(2.1rem,5vw,4rem);line-height:1.03;margin:0 0 18px;max-width:900px}} h2{{margin-top:0;font-size:1.65rem}} h3{{margin-bottom:8px}}
.lede{{max-width:780px;font-size:1.16rem;color:#e3eef2}} .eyebrow{{letter-spacing:.12em;text-transform:uppercase;font-weight:700;color:#9fe0d8}}
main{{padding:34px 24px 70px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin:24px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;box-shadow:0 5px 18px rgba(18,48,74,.05)}}
.kpi{{font-size:2.25rem;font-weight:800;color:var(--navy)}} .kpi-label{{color:var(--muted)}} section{{margin:34px 0}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}}
.bar-row{{display:grid;grid-template-columns:minmax(120px,1.4fr) 2fr 40px 52px;align-items:center;gap:10px;margin:11px 0;font-size:.9rem}}
.track{{height:11px;background:#e8eff0;border-radius:99px;overflow:hidden}} .fill{{height:100%;background:linear-gradient(90deg,var(--teal),#48b5a9);border-radius:99px}}
.bar-row small{{color:var(--muted)}} table{{width:100%;border-collapse:collapse;background:white}} th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line);vertical-align:top}} th{{background:#edf4f3;color:var(--navy)}}
.callout{{border-left:5px solid var(--gold);background:#fff7e6;padding:18px 20px;border-radius:4px}} .pass{{display:inline-block;padding:5px 10px;border-radius:99px;background:#dff4ec;color:#12634f;font-weight:800}}
code{{background:#eef1f3;padding:.15em .35em;border-radius:4px}} a{{color:#006b70}} footer{{color:var(--muted);font-size:.9rem;margin-top:48px}}
@media(max-width:620px){{.bar-row{{grid-template-columns:1fr 2fr 34px}}.bar-row small{{display:none}}table{{font-size:.85rem}}}}
</style></head><body>
<header><div class="wrap"><div class="eyebrow">Independent portfolio research</div><h1>UK Public Sector Algorithm Transparency Audit</h1>
<p class="lede">A reproducible audit of what UK public bodies disclose about algorithmic tools in the official GOV.UK repository. The analysis measures transparency coverage, not safety, legality, fairness, or compliance.</p></div></header>
<main class="wrap">
<div class="grid"><div class="card"><div class="kpi">{total}</div><div class="kpi-label">official records analysed</div></div>
<div class="card"><div class="kpi">{len(organisations)}</div><div class="kpi-label">publishing organisations</div></div>
<div class="card"><div class="kpi">{summary['mean_disclosure_score']:.1f}</div><div class="kpi-label">mean disclosure coverage / 100</div></div>
<div class="card"><div class="kpi">{html.escape(latest)}</div><div class="kpi-label">latest publication date</div></div></div>
<section class="callout"><strong>Read the score correctly.</strong> A higher score means more selected governance topics had substantive public text. It does not prove that the underlying controls are effective, and a lower score may reflect ATRS version differences, legitimate non-applicability, or publication choices.</section>
<section><h2>Governance disclosure coverage</h2><div class="card" style="overflow:auto"><table><thead><tr><th>Indicator</th><th>Coverage</th><th>Records</th><th>What was checked</th></tr></thead><tbody>{coverage_rows}</tbody></table></div></section>
<section class="two"><div class="card"><h2>Deployment phase</h2>{phase_bars}</div><div class="card"><h2>Most represented organisations</h2>{top_orgs}</div></section>
<section><div class="card"><h2>Public functions represented</h2>{function_bars}</div></section>
<section class="two"><div class="card"><h2>Data quality</h2><p><span class="pass">{qa_status}</span></p><ul>
<li>Source reports {checks['reported_total']} records; {checks['records_fetched']} were fetched.</li>
<li>Duplicate source URLs: {checks['duplicate_source_urls']}.</li><li>Missing titles: {checks['missing_title']}.</li><li>Missing organisations: {checks['missing_organisation']}.</li>
</ul></div><div class="card"><h2>Method boundary</h2><p>The pipeline detects substantive text in eight governance themes using heading aliases across ATRS versions. It does not use sentiment analysis or infer control effectiveness from prose.</p><p>Review <a href="../docs/METHODOLOGY.md">the methodology</a> and <a href="../docs/DATA_CARD.md">data card</a> before reusing the figures.</p></div></section>
<footer>Source: <a href="{html.escape(source['source_url'])}">GOV.UK Algorithmic Transparency Records</a>. Retrieved {html.escape(source['retrieved_at_utc'])}. Source content is available under the Open Government Licence v3.0. This independent analysis is not endorsed by the UK Government.</footer>
</main><script type="application/json" id="audit-summary">{html.escape(json.dumps(summary))}</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return summary
