"""
api-server operation: a small HTTP wrapper around the Downloader.

Exposes a single endpoint, `POST /download`, that accepts a Spotify URL and
runs the regular spotdl download pipeline with the server's configured
settings. When the URL is a Spotify *playlist*, the server additionally
generates an `.m3u8` playlist file alongside the downloaded audio so that
Navidrome (or any other tag-aware library) can ingest it.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from spotdl.download.downloader import Downloader
from spotdl.types.options import DownloaderOptions, WebOptions
from spotdl.utils.search import get_simple_songs

__all__ = ["api_server"]

logger = logging.getLogger(__name__)

_KNOWN_KINDS = {"track", "album", "playlist", "artist"}


class DownloadRequest(BaseModel):
    """Body of POST /download."""

    url: str


class DownloadResponse(BaseModel):
    """Returned by POST /download once the pipeline finishes."""

    url: str
    kind: str
    downloaded: List[str]
    errors: List[str]


def api_server(
    downloader_settings: DownloaderOptions,
    server_settings: WebOptions,
) -> None:
    """
    Run the api-server. Blocks the calling thread until the process is killed.

    ### Arguments
    - downloader_settings: defaults applied to every download. The server adds
      a per-request `m3u` override when the URL is a playlist.
    - server_settings: WebOptions used for host/port (we reuse the same flags
      that `spotdl web` already wires up).
    """

    app = FastAPI(title="spotdl api-server")
    # Serialize downloads. spotdl's Downloader and a few of its module-level
    # caches (yt-dlp temp dir, archive file, m3u write) are not safe to run
    # concurrently within a single process.
    download_lock = threading.Lock()

    @app.post("/download", response_model=DownloadResponse)
    def download(req: DownloadRequest) -> DownloadResponse:
        if not download_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409, detail="another download is in progress"
            )
        try:
            return _run_download(req.url, downloader_settings)
        finally:
            download_lock.release()

    host = server_settings.get("host") or "0.0.0.0"
    port = int(server_settings.get("port") or 8800)
    logger.info("spotdl api-server listening on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


def _run_download(
    url: str, base_settings: DownloaderOptions
) -> DownloadResponse:
    kind = _kind_from_url(url)
    per_request: Dict[str, Any] = dict(base_settings)

    m3u_template = _m3u_template_for(url, base_settings["output"])
    if m3u_template:
        per_request["m3u"] = m3u_template
        # spotdl writes the m3u with a plain open() — it doesn't create parent
        # dirs. Pre-create the playlist directory so the first run succeeds.
        Path(m3u_template.split("{", 1)[0]).expanduser().mkdir(
            parents=True, exist_ok=True
        )

    downloader = Downloader(per_request)
    try:
        try:
            songs = get_simple_songs(
                [url],
                use_ytm_data=downloader.settings["ytm_data"],
                playlist_numbering=downloader.settings["playlist_numbering"],
                albums_to_ignore=downloader.settings["ignore_albums"],
                album_type=downloader.settings["album_type"],
                playlist_retain_track_cover=downloader.settings[
                    "playlist_retain_track_cover"
                ],
            )
        except Exception as exc:
            logger.exception("Failed to resolve %s", url)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        results = downloader.download_multiple_songs(songs)

        downloaded = [
            str(path) for _song, path in results if path is not None
        ]
        return DownloadResponse(
            url=url,
            kind=kind,
            downloaded=downloaded,
            errors=list(downloader.errors),
        )
    finally:
        downloader.progress_handler.close()


def _kind_from_url(url: str) -> str:
    """Best-effort categorization of a Spotify URL."""

    try:
        path = urlparse(url).path.lstrip("/")
        head = path.split("/", 1)[0] if path else ""
        if head in _KNOWN_KINDS:
            return head
    except Exception:  # pylint: disable=broad-except
        pass
    return "unknown"


def _m3u_template_for(url: str, output_template: str) -> Optional[str]:
    """
    Return an m3u path template only when the URL is a Spotify playlist. The
    m3u lands in a `Playlists/` subdirectory next to the audio files; we walk
    the output template up to its concrete prefix to find that root.
    """

    if _kind_from_url(url) != "playlist":
        return None

    # The token-free prefix of the output template is the root music dir.
    # Example: "/music/{album-artist} - {title}.{output-ext}" → "/music".
    base = output_template.split("{", 1)[0].rstrip("/") or "."
    return f"{base}/Playlists/{{list[0]}}.m3u8"
