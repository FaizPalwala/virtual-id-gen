# SFHQ Part 1 — Licence Evidence (human-readable extract)

**Source:** https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1
**Captured:** 2026-08-03 (live fetch, `curl` with browser UA)
**Raw HTML:** `kaggle_sfhq_part1.html` (14,648 bytes) — sha256:
`e99770b64f2d3727392489e2faa88cb1076b5bc02df197cde6b63f441ff61eb3`
**Kaggle API JSON:** `kaggle_sfhq_part1.api.json` (first-party structured record)

> **Why the raw HTML looks blank in a browser:** Kaggle serves a
> JavaScript single-page app.  The licence metadata lives in the page's
> inline schema.org JSON-LD inside `<head>`; the visible body is rendered
> client-side and never runs when the file is opened locally.  The JSON-LD
> below is the authoritative record and is present verbatim in the raw file.
> The preferred evidence is the **Kaggle API JSON** (first-party, machine-
> readable) documented at the bottom of this file.

## Verbatim schema.org JSON-LD (from the saved HTML)

```json
"license":{"@type":"CreativeWork","name":"CC0: Public Domain","url":"https://creativecommons.org/publicdomain/zero/1.0/"}
```

## Associated metadata (same JSON-LD block)

| Field | Value |
|---|---|
| Dataset name | Synthetic Faces High Quality (SFHQ) part 1 |
| Kaggle dataset id | 2437902 |
| Creator | David Beniaguev (selfishgene) |
| Content | 90K curated 1024x1024 face images. StyleGAN2 encoding of paintings and 3D models |
| Date modified | 2022-12-17 |
| isAccessibleForFree | true |
| File size | 14,855,578,693 bytes (zip, requires Kaggle subscription to download) |
| Alternate name | 90K curated 1024x1024 face images. StyleGAN2 encoding of paintings and 3D models |

## First-party API record (preferred evidence)

The Kaggle public API returns the licence as structured data.  This is the
same API Kaggle itself uses to render the page — machine-readable,
verifiable, and re-fetchable by any reviewer:

```bash
curl -sL "https://www.kaggle.com/api/v1/datasets/view/selfishgene/synthetic-faces-high-quality-sfhq-part-1"
```

Key fields from the saved `kaggle_sfhq_part1.api.json` (fetched 2026-08-03):

```json
{
  "licenseNameNullable": "CC0: Public Domain",
  "id": 2437902,
  "creatorNameNullable": "David Beniaguev",
  "title": "Synthetic Faces High Quality (SFHQ) part 1",
  "subtitle": "90K curated 1024x1024 face images. StyleGAN2 encoding of paintings and 3D models",
  "totalBytes": 15045664192,
  "isPrivate": false
}
```

## Verification

```bash
# Confirm the raw file matches this record:
shasum -a 256 kaggle_sfhq_part1.html
# e99770b64f2d3727392489e2faa88cb1076b5bc02df197cde6b63f441ff61eb3

# Confirm the CC0 license string is present in the raw HTML:
grep -o '"name":"CC0: Public Domain"' kaggle_sfhq_part1.html
# "name":"CC0: Public Domain"

# Confirm the API record (re-fetchable by any reviewer):
curl -sL "https://www.kaggle.com/api/v1/datasets/view/selfishgene/synthetic-faces-high-quality-sfhq-part-1" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['licenseNameNullable'])"
# CC0: Public Domain
```

## Interpretation

The Kaggle listing for SFHQ Part 1 declares its licence as **CC0 / Public
Domain** (Creative Commons Zero, https://creativecommons.org/publicdomain/zero/1.0/).
The upstream GitHub repo's MIT licence covers only the *code*, not the image
data; the Kaggle listing is the licence evidence for the data itself.  This
is the file to keep on record per Strat.md §2.1.
