"""Minimal client for the official MyAnimeList API v2.

Standalone port of AniOntology's ingestion/myanimelist/client.py, trimmed to
only what AniRestore needs (fetching cover-art / main_picture data) and with
no import dependency on the AniOntology codebase, so this module can be
copied or reused independently by either project.
"""

import logging
import time

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.mal.config import settings
from src.mal.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_MAL_FIELDS = "id,title,alternative_titles,main_picture"


class MALAPIError(Exception):
    pass


class MALClient:
    """Client for the official MyAnimeList API v2."""

    def __init__(self) -> None:
        if not settings.mal_client_id:
            raise ValueError("MAL_CLIENT_ID is not set (env var or .env file)")
        self._rate_limiter = RateLimiter(calls_per_period=settings.mal_rpm, period_seconds=60.0)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-MAL-CLIENT-ID": settings.mal_client_id,
                "Accept": "application/json",
            }
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _get(self, path: str, params: dict | None = None) -> dict:
        self._rate_limiter.wait()
        url = f"{settings.mal_api_url}/{path.lstrip('/')}"
        resp = self._session.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10))
            logger.warning("mal rate limited, sleeping %ss", retry_after)
            time.sleep(retry_after)
            resp = self._session.get(url, params=params, timeout=15)
        if not resp.ok:
            raise MALAPIError(f"MAL API error {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def get_anime_by_id(self, mal_id: int, fields: str = _MAL_FIELDS) -> dict:
        return self._get(f"anime/{mal_id}", {"fields": fields})
