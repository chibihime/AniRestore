"""Download anime cover-art images listed in a cover manifest.

Reads the manifest produced by fetch_cover_manifest.py (mal_id -> image_url)
and downloads each image into data/raw_covers/{mal_id}.{ext}. Idempotent:
already-downloaded, valid images are skipped on re-run. Failures (network
errors, corrupt/undersized images) are logged and written to a report file
rather than aborting the whole run.

Usage:
    python src/download_covers.py \
        --manifest data/cover_manifest.json \
        --out-dir data/raw_covers
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from src.mal.rate_limiter import RateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_SIDE_PX = 64


def _ext_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in (".jpg", ".jpeg", ".png", ".webp") else ".jpg"


def _validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
                return False
        return True
    except Exception:
        return False


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
def _fetch(session: requests.Session, url: str) -> bytes:
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return resp.content


def download_one(entry: dict, out_dir: Path, session: requests.Session, limiter: RateLimiter) -> dict:
    mal_id = entry["mal_id"]
    url = entry["image_url"]
    dest = out_dir / f"{mal_id}{_ext_from_url(url)}"

    if dest.exists() and _validate_image(dest):
        return {"mal_id": mal_id, "status": "skipped"}

    limiter.wait()
    try:
        content = _fetch(session, url)
    except Exception as e:
        return {"mal_id": mal_id, "status": "failed", "error": f"download error: {e}"}

    dest.write_bytes(content)
    if not _validate_image(dest):
        dest.unlink(missing_ok=True)
        return {"mal_id": mal_id, "status": "failed", "error": "corrupt or undersized image"}

    return {"mal_id": mal_id, "status": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cover_manifest.json")
    parser.add_argument("--out-dir", default="data/raw_covers")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rps", type=float, default=10.0, help="max requests/sec to the CDN")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    logger.info("loaded %d entries from %s", len(entries), manifest_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    limiter = RateLimiter(calls_per_period=args.rps, period_seconds=1.0)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, e, out_dir, session, limiter) for e in entries]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 100 == 0:
                logger.info("processed %d/%d", i, len(entries))

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures = []
    for r in results:
        counts[r["status"]] += 1
        if r["status"] == "failed":
            failures.append(r)

    report_path = out_dir.parent / "download_failures.json"
    report_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    logger.info(
        "done: %d downloaded, %d skipped (already present), %d failed (see %s)",
        counts["downloaded"],
        counts["skipped"],
        counts["failed"],
        report_path,
    )


if __name__ == "__main__":
    main()
