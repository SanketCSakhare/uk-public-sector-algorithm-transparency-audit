# Data card

## Dataset

UK Algorithmic Transparency Records (ATRS), published on GOV.UK by the Cabinet Office, Department for Science, Innovation and Technology, and Government Digital Service.

## Source of truth

- Finder: <https://www.gov.uk/algorithmic-transparency-records>
- Search API: <https://www.gov.uk/api/search.json>
- Record API pattern: `https://www.gov.uk/api/content/algorithmic-transparency-records/{slug}`

## Licence

Open Government Licence v3.0, except where GOV.UK identifies third-party material.

## Grain and keys

- Grain: one published ATRS record
- Stable source identifier: GOV.UK content ID
- Candidate key: source URL

## Main fields

Title, description, organisation, organisation type, public function, capability, deployment phase, region, publication date, raw and normalised ATRS versions, eight disclosure flags, field counts, and disclosure coverage score.

## Processing

The repository stores a compact official metadata snapshot and a derived record-level CSV. Full source prose is not republished. The pipeline refetches current record bodies when rerun and derives disclosure flags from field headings and substantive text.

## Known issues

- ATRS schema changes across versions.
- Some fields are legitimately not applicable.
- Organisation slugs are retained to preserve source vocabulary.
- Source updates may change historical rows between snapshots.
- GOV.UK currently contains several labels for the same ATRS version (for example `v3`, `3.0`, and `v3.0`); both the raw and normalised values are retained.

## Ethical considerations

The project avoids inferring safety or compliance, does not rank organisations, and does not republish contact emails or full record text. Each row links back to its official source for contextual reading.
