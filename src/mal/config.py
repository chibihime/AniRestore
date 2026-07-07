"""MAL pipeline settings, loaded from environment / .env.

Reuses the same env var names as AniOntology's utils/config.py
(MAL_CLIENT_ID, MAL_RPM) so a single MAL API app/client ID can be shared
across both projects without duplicating credentials.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    mal_api_url: str = os.environ.get("MAL_API_URL", "https://api.myanimelist.net/v2")
    mal_client_id: str = os.environ.get("MAL_CLIENT_ID", "")
    mal_rpm: int = int(os.environ.get("MAL_RPM", "90"))

    # Path to AniOntology's cached raw MAL dataset, used to derive the same
    # set of anime IDs so the two projects' cover-art datasets stay linked.
    aniontology_raw_json: str = os.environ.get(
        "ANIONTOLOGY_RAW_JSON",
        "../AniOntology/data/raw/myanimelist_anime.json",
    )


settings = Settings()
