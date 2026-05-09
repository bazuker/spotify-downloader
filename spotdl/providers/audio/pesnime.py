"""
pesni.me module for searching and downloading songs.
"""

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

import requests

from spotdl.providers.audio.base import AudioProvider
from spotdl.types.result import Result
from spotdl.utils.config import GlobalConfig

__all__ = ["PesniMe"]

logger = logging.getLogger(__name__)

BASE_URL = "https://music.pesni.me/"

# Last-known Next.js Server Action ID. Used as a fallback only when we cannot
# scrape a fresh ID from the live site. See `_resolve_action_id` for details.
FALLBACK_ACTION_ID = "601857be76a91a90699eab9739b49e33b38ca368ff"

# pesni.me names its search server action "searchSuggestsAction" in the client
# bundle. We grep for that name to find the matching action ID hash, which is
# what the `next-action` header has to carry.
ACTION_FUNCTION_NAME = "searchSuggestsAction"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

POST_HEADERS = {
    "Accept": "text/x-component",
    "Content-Type": "text/plain;charset=UTF-8",
    **BROWSER_HEADERS,
}

# Only inspect the first few tracks pesni.me returns — its top hits are
# already very tight; scanning further just produces noise.
MAX_CANDIDATES = 5

_CHUNK_SRC_PATTERN = re.compile(r'src="(/_next/static/chunks/[^"]+\.js)"')

# Minified Next.js bundles wrap the call as `(0,X.createServerReference)("HASH",
# ...args..., "FUNCTION_NAME")`, so we anchor on the hash + function-name pair
# instead of the function call itself. Between the two string literals the
# bundler only emits identifiers and commas — no other quoted strings — so
# the negated-class lookahead is safe.
_ACTION_REF_PATTERN = re.compile(
    r'"([0-9a-f]{40,64})"\s*,[^"]*"' + ACTION_FUNCTION_NAME + r'"'
)

_action_id_lock = threading.Lock()
_action_id_cache: Optional[str] = None


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

        action_id = _resolve_action_id() or FALLBACK_ACTION_ID
        payload = _post_search(search_term, action_id)

        # No payload usually means pesni.me redeployed and the action ID we
        # just used is stale (the server returns the home-page HTML instead of
        # an RSC stream). Re-scrape and try once more.
        if payload is None:
            _invalidate_action_id()
            fresh_id = _resolve_action_id()
            if fresh_id and fresh_id != action_id:
                payload = _post_search(search_term, fresh_id)

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


def _post_search(search_term: str, action_id: str) -> Optional[Dict[str, Any]]:
    """
    POST the search term to pesni.me with the given action ID and parse the
    RSC response. Returns the decoded payload, or None on any failure.
    """

    headers = {**POST_HEADERS, "next-action": action_id}
    try:
        response = requests.post(
            BASE_URL,
            headers=headers,
            data=json.dumps([search_term], ensure_ascii=False).encode("utf-8"),
            timeout=10,
            proxies=GlobalConfig.get_parameter("proxies"),
        )
        response.raise_for_status()
    except Exception as exc:
        logger.debug("pesni.me search failed for %s: %s", search_term, exc)
        return None

    # pesni.me sends UTF-8 but does not advertise a charset, so requests
    # falls back to ISO-8859-1 and mangles Cyrillic. Decode explicitly.
    body = response.content.decode("utf-8", errors="replace")
    return _parse_action_payload(body)


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


def _resolve_action_id() -> Optional[str]:
    """
    Scrape pesni.me's home page to discover the current search action ID.

    The ID is a build-time hash that rotates on every redeploy. It lives in
    one of the page's JS chunks inside a `createServerReference("HASH", ...,
    "searchSuggestsAction")` call. We cache the resolved ID for the lifetime
    of the process and only re-scrape when the cache is invalidated (e.g.
    after a search returns no payload).
    """

    global _action_id_cache  # noqa: PLW0603

    # Hold the lock for the entire scrape so concurrent callers (spotdl
    # downloads tracks in a thread pool) don't all stampede the network on a
    # cold cache. The first thread does the work; the rest wait briefly and
    # then read the cached value.
    with _action_id_lock:
        if _action_id_cache:
            return _action_id_cache

        try:
            html = requests.get(BASE_URL, headers=BROWSER_HEADERS, timeout=10).text
        except Exception as exc:
            logger.debug("pesni.me homepage fetch failed: %s", exc)
            return None

        chunk_paths = sorted(set(_CHUNK_SRC_PATTERN.findall(html)))
        base = BASE_URL.rstrip("/")
        for path in chunk_paths:
            try:
                chunk = requests.get(
                    base + path, headers=BROWSER_HEADERS, timeout=10
                ).text
            except Exception as exc:
                logger.debug("pesni.me chunk fetch failed for %s: %s", path, exc)
                continue

            match = _ACTION_REF_PATTERN.search(chunk)
            if not match:
                continue

            _action_id_cache = match.group(1)
            logger.debug(
                "pesni.me: resolved action id %s from %s", _action_id_cache, path
            )
            return _action_id_cache

        logger.debug(
            "pesni.me: action id not found in any of %d chunks", len(chunk_paths)
        )
        return None


def _invalidate_action_id() -> None:
    """Clear the cached action ID so the next resolve attempt re-scrapes."""

    global _action_id_cache  # noqa: PLW0603

    with _action_id_lock:
        _action_id_cache = None
