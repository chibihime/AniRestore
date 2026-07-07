"""Derive a cover-art manifest (mal_id -> image_url) for AniRestore.

Reads the same anime IDs already ingested by AniOntology's MAL pipeline (its
cached data/raw/myanimelist_anime.json) so both projects stay linked to the
same underlying dataset, rather than re-fetching the full anime list from
scratch. Any entries missing a cached image_url are backfilled with a live
call to the MAL API via src.mal.client.MALClient.

Usage:
    python src/fetch_cover_manifest.py \
        --source ../AniOntology/data/raw/myanimelist_anime.json \
        --out data/cover_manifest.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.mal.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_source_entries(source_path: Path) -> list[dict]:
    with open(source_path, "r", encoding="utf-8") as f:
        return json.load(f)


def backfill_missing(entries: list[dict]) -> None:
    """Mutates entries in place, filling image_url via a live MAL API call."""
    missing = [e for e in entries if not e.get("image_url")]
    if not missing:
        return

    from src.mal.client import MALClient, MALAPIError

    logger.info("backfilling %d entries missing image_url via MAL API", len(missing))
    client = MALClient()
    for entry in missing:
        mal_id = entry["mal_id"]
        try:
            raw = client.get_anime_by_id(mal_id)
            entry["image_url"] = (raw.get("main_picture") or {}).get("large")
            if not entry["image_url"]:
                logger.warning("mal_id=%s has no main_picture even after backfill", mal_id)
        except MALAPIError as e:
            logger.warning("mal_id=%s backfill failed: %s", mal_id, e)


def build_manifest(entries: list[dict]) -> list[dict]:
    manifest = []
    for e in entries:
        if not e.get("image_url"):
            continue
        manifest.append(
            {
                "mal_id": e["mal_id"],
                "title": e.get("title_romaji") or e.get("title_english"),
                "image_url": e["image_url"],
            }
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=settings.aniontology_raw_json)
    parser.add_argument("--out", default="data/cover_manifest.json")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(
            "source dataset not found at %s — pass --source pointing at "
            "AniOntology's data/raw/myanimelist_anime.json",
            source_path,
        )
        sys.exit(1)

    entries = load_source_entries(source_path)
    logger.info("loaded %d anime entries from %s", len(entries), source_path)

    backfill_missing(entries)
    manifest = build_manifest(entries)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "wrote %d/%d entries with a resolved image_url to %s",
        len(manifest),
        len(entries),
        out_path,
    )


if __name__ == "__main__":
    main()
