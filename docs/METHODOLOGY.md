# Methodology

## Research question

What governance topics are substantively addressed in the public text of records published through the UK Algorithmic Transparency Recording Standard (ATRS)?

## Unit of analysis

One published record in the official GOV.UK Algorithmic Transparency Records finder. The source URL and GOV.UK content ID form the record identity.

## Collection

The pipeline queries the GOV.UK Search API for all documents whose type is `algorithmic_transparency_record`. It requests a limited set of metadata fields, reconciles the returned count to the source-reported total, then retrieves each record through the GOV.UK Content API. Retrieval uses bounded concurrency, timeouts, retries, and a descriptive user agent.

## Parsing

ATRS versions differ. The GOV.UK Content API provides record bodies as semantic HTML. The parser treats every level-three heading as a field and captures the text until the next level-two or level-three heading. Duplicate heading identifiers are preserved with deterministic suffixes.

## Indicators

The analysis uses heading aliases, not unrestricted keyword searches over an entire record. A matching field must contain at least 12 meaningful characters and cannot consist solely of a common missing-value marker.

| Indicator | Evidence sought |
|---|---|
| Named accountability | senior responsible owner or contact field |
| Human oversight | human review, oversight, or decision responsibility |
| Appeal or review route | appeal, review, or contestability field |
| Impact assessment | impact-assessment field |
| Risks and mitigations | risk-and-mitigation field |
| Performance evidence | model/tool performance or accuracy field |
| Operational data governance | data sources, access/storage, or sharing field |
| Monitoring or maintenance | maintenance, monitoring, or ongoing-review field |

The disclosure coverage score is the unweighted percentage of these eight indicators that pass. Equal weighting is a communication choice, not a validated policy model.

## Quality gates

The live run checks:

- fetched record count equals the GOV.UK-reported total;
- source URLs are unique;
- titles, organisations, publication dates, and ATRS versions are profiled for missingness;
- deployment phases conform to the published finder vocabulary; and
- all detail-page fetches succeed.

Count mismatch and duplicate source URLs are blocking failures. Other issues are preserved in `reports/data_quality.json` for interpretation.

## Limitations

1. The audit measures disclosure presence, not truth, quality, or control effectiveness.
2. Older and newer ATRS versions are not structurally identical, so scores are not directly comparable without version-aware review.
3. A legitimate `N/A` can reduce coverage even when the field does not apply.
4. Public records cannot describe undisclosed tools or exempt information.
5. Heading aliases may miss unusual wording and can produce false negatives.
6. Publication dates reflect repository records, not necessarily deployment dates.

## Appropriate use

Use the output to explore the repository, identify themes for qualitative review, and demonstrate a reproducible governance-analysis method. Do not use it to produce organisational league tables, certify compliance, or claim that a disclosed control works.
