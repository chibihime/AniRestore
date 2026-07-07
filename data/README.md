# Dataset: Anime Cover Art

## Source and linkage to AniOntology

Cover-art URLs are **not** re-fetched from the live MAL API. They're derived from
AniOntology's already-ingested `data/raw/myanimelist_anime.json` (its MAL
ingestion pipeline populates `image_url` from each anime's `main_picture.large`
field). Reusing that cached dataset keeps the two projects pointed at the
exact same set of MAL anime IDs, and avoids spending MAL API rate-limit budget
re-fetching data AniOntology already has.

`src/mal/client.py` still provides a standalone MAL API v2 client (same
auth/rate-limit approach as AniOntology's, but with no import dependency on
that codebase) — `src/fetch_cover_manifest.py` uses it only to backfill any
entries that are missing a cached `image_url`. On the current dataset, 0/2000
entries required backfill.

## Pipeline

1. `python -m src.fetch_cover_manifest` — reads AniOntology's cached MAL
   dataset, writes `data/cover_manifest.json` (list of `{mal_id, title,
   image_url}`).
2. `python -m src.download_covers` — downloads each `image_url` into
   `data/raw_covers/{mal_id}.{ext}`, skipping files that already exist and
   validate correctly (idempotent re-runs), and validating every downloaded
   image (`PIL.Image.verify()` + a 64px minimum side) before accepting it.
   Failures are written to `data/download_failures.json` instead of aborting
   the run.
3. `src.data_loader.CoverArtDataset` / `load_image_tensor` load images from
   `data/raw_covers/` into normalized `[0, 1]` float32 CHW tensors, with an
   optional `resize_to` for uniform-shape batches.

## Current dataset size

- **2000/2000** anime entries from AniOntology's dataset resolved to a valid
  `image_url` (0 required live-API backfill).
- **2000/2000** images downloaded successfully — **0 failures**.
- Formats: 1323 `.jpg`, 677 `.webp` (as served by MAL's CDN — both are
  standard Pillow-decodable formats, no conversion needed at download time).
- Total size on disk: ~143 MB.

## Exclusions

None yet — every entry in the source dataset had a resolvable cover image and
downloaded/validated cleanly. If future re-runs (e.g. after AniOntology
re-ingests a larger MAL page range) introduce broken/undersized images, they
will show up in `data/download_failures.json` rather than silently breaking
the dataset.

## Re-running

`data/raw_covers/`, `data/cover_manifest.json`, and `data/download_failures.json`
are gitignored (large/regeneratable binary data) — regenerate locally via the
two commands above. Requires `ANIONTOLOGY_RAW_JSON` (see `.env.example`) to
point at a checked-out AniOntology repo with its MAL data already ingested.
