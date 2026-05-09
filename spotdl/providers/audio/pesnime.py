"""
pesni.me module for searching and downloading songs.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import requests

from spotdl.providers.audio.base import AudioProvider
from spotdl.types.result import Result
from spotdl.utils.config import GlobalConfig

__all__ = ["PesniMe"]

logger = logging.getLogger(__name__)

BASE_URL = "https://music.pesni.me/"

# Next.js Server Action ID for pesni.me's search action. This is a build-time
# hash and will rotate whenever pesni.me redeploys; refresh it from the network
# tab if searches start returning empty.
NEXT_ACTION_ID = "601857be76a91a90699eab9739b49e33b38ca368ff"

HEADERS = {
    "Accept": "text/x-component",
    "Content-Type": "text/plain;charset=UTF-8",
    "next-action": NEXT_ACTION_ID,
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Only inspect the first few tracks pesni.me returns — its top hits are
# already very tight; scanning further just produces noise.
MAX_CANDIDATES = 5


class PesniMe(AudioProvider):
    """
    pesni.me audio provider class.

    Returns only results whose "<artist> - <title>" pair exactly matches the
    requested search term. Anything less than an exact match falls through to
    the next provider in the chain.
    """

    SUPPORTS_ISRC = False
    GET_RESULTS_OPTS: List[Dict[str, Any]] = [{}]

    def get_results(self, search_term: str, *_args, **_kwargs) -> List[Result]:
        """
        Search pesni.me for the given "<artist> - <title>" term.

        ### Arguments
        - search_term: The "<artist> - <title>" string to search for.

        ### Returns
        - A list of pesni.me results that exactly match the search term.
        """

        try:
            response = requests.post(
                BASE_URL,
                headers=HEADERS,
                data=json.dumps([search_term], ensure_ascii=False).encode("utf-8"),
                timeout=10,
                proxies=GlobalConfig.get_parameter("proxies"),
            )
            response.raise_for_status()
        except Exception as exc:
            logger.debug("pesni.me search failed for %s: %s", search_term, exc)
            return []

        # pesni.me sends UTF-8 but does not advertise a charset, so requests
        # falls back to ISO-8859-1 and mangles Cyrillic. Decode explicitly.
        body = response.content.decode("utf-8", errors="replace")
        payload = _parse_action_payload(body)
        if payload is None:
            logger.debug("pesni.me returned no payload for %s", search_term)
            return []

        items = (payload.get("tracks") or {}).get("items") or []
        normalized_query = search_term.strip().lower()

        results: List[Result] = []
        for item in items[:MAX_CANDIDATES]:
            artist = (item.get("artist") or "").strip()
            title = (item.get("title") or "").strip()
            if not artist or not title:
                continue

            if f"{artist} - {title}".lower() != normalized_query:
                continue

            download_url = item.get("download") or item.get("play")
            if not download_url:
                continue

            results.append(
                Result(
                    source="pesni.me",
                    url=download_url,
                    # Self-verified: we only accept exact "<artist> - <title>" matches,
                    # so the base class can short-circuit on this result.
                    verified=True,
                    name=title,
                    duration=float(item.get("duration") or 0),
                    author=artist,
                    result_id=str(item.get("id") or download_url),
                    search_query=search_term,
                    artists=tuple(a.strip() for a in artist.split(",") if a.strip()),
                )
            )

        return results


def _parse_action_payload(body: str) -> Optional[Dict[str, Any]]:
    """
    pesni.me's endpoint replies with a Next.js RSC-style stream where each
    line is "<id>:<json>". Only the line with id "1" carries the search
    payload; the line with id "0" is a metadata frame we discard.
    """

    for line in body.splitlines():
        head, sep, rest = line.partition(":")
        if not sep or head.strip() != "1":
            continue
        try:
            return json.loads(rest)
        except json.JSONDecodeError:
            return None

    return None
