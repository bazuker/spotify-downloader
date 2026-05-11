"""
api-server operation: HTTP wrapper around the Downloader that runs downloads
as background jobs.

Workflow:
  1. `POST /download {url}` → returns `{job_id, state}` immediately (202).
  2. `GET /jobs/{job_id}` → returns the current `{state, total, downloaded,
     errored, skipped, paths, error}`. Poll until `state` is `done`/`failed`.

Single-flight: at most one non-terminal job exists at any time. New
submissions while one is in progress get 409.

State is in-memory only. A restart loses any in-flight or recently
completed jobs — for the personal-bot use case that's acceptable; resubmit
the URL.
"""

import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from spotdl.download.downloader import LYRICS_PROVIDERS, Downloader
from spotdl.types.options import DownloaderOptions, WebOptions
from spotdl.types.song import Song
from spotdl.utils.archive import Archive
from spotdl.utils.formatter import create_file_name
from spotdl.utils.metadata import embed_metadata
from spotdl.utils.search import get_simple_songs

__all__ = ["api_server"]

logger = logging.getLogger(__name__)

_KNOWN_KINDS = {"track", "album", "playlist", "artist"}
_ARCHIVE_FILENAME = "musicdon.archive"

# Serializes archive read-modify-write across endpoints. The Downloader holds
# its own copy of the archive in memory during a /download run and saves at
# the end; an /enrich finishing mid-download would otherwise race that save.
_archive_lock = threading.Lock()

# How long terminal (done/failed) jobs stay readable after they finish. Long
# enough that the bot's poll loop can pick up the final state and include it
# in its Telegram message; short enough that the in-memory map doesn't bloat
# over weeks of uptime.
JOB_RETENTION_SECONDS = 600

STATE_QUEUED = "queued"
STATE_RESOLVING = "resolving"
STATE_DOWNLOADING = "downloading"
STATE_DONE = "done"
STATE_FAILED = "failed"
_TERMINAL_STATES = {STATE_DONE, STATE_FAILED}


class SubmitRequest(BaseModel):
    """Body of POST /download."""

    url: str


class EnrichResponse(BaseModel):
    """Returned by POST /enrich."""

    url: str
    output_path: str
    song_name: str
    song_artist: str


class SubmitResponse(BaseModel):
    """Returned by POST /download — the bot polls /jobs/{job_id} from here on."""

    job_id: str
    state: str


class JobStatus(BaseModel):
    """Returned by GET /jobs/{job_id}."""

    job_id: str
    url: str
    kind: str
    state: str
    total: Optional[int] = None
    downloaded: int = 0
    errored: int = 0
    skipped: int = 0
    paths: List[str] = []
    errored_tracks: List[str] = []
    error: Optional[str] = None


@dataclass
class Job:
    """In-memory progress record for one download."""

    id: str
    url: str
    kind: str = "unknown"
    state: str = STATE_QUEUED
    total: Optional[int] = None
    downloaded: int = 0
    errored: int = 0
    skipped: int = 0  # already-archived tracks; not counted against `total`
    paths: List[str] = field(default_factory=list)
    errored_tracks: List[str] = field(default_factory=list)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES


class JobStore:
    """
    In-memory job registry with single-flight enforcement and TTL cleanup.

    Single-flight: at most one non-terminal job at a time. New submissions
    while one is active raise ValueError, which the endpoint surfaces as
    HTTP 409.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._active_id: Optional[str] = None

    def submit(self, url: str) -> Job:
        with self._lock:
            self._gc_locked()
            if self._active_id is not None:
                active = self._jobs.get(self._active_id)
                if active is not None and not active.is_terminal():
                    raise ValueError("another download is in progress")
            job_id = uuid.uuid4().hex[:12]
            job = Job(id=job_id, url=url)
            self._jobs[job_id] = job
            self._active_id = job_id
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            self._gc_locked()
            return self._jobs.get(job_id)

    def _gc_locked(self) -> None:
        now = time.time()
        stale = [
            jid
            for jid, j in self._jobs.items()
            if j.is_terminal()
            and j.finished_at is not None
            and (now - j.finished_at) > JOB_RETENTION_SECONDS
        ]
        for jid in stale:
            del self._jobs[jid]
            if self._active_id == jid:
                self._active_id = None


def api_server(
    downloader_settings: DownloaderOptions,
    server_settings: WebOptions,
) -> None:
    """
    Run the api-server. Blocks the calling thread until the process is killed.
    """

    _apply_dedup_defaults(downloader_settings)
    jobs = JobStore()

    app = FastAPI(title="spotdl api-server")

    @app.post("/download", response_model=SubmitResponse, status_code=202)
    def submit(req: SubmitRequest) -> SubmitResponse:
        try:
            job = jobs.submit(req.url)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        threading.Thread(
            target=_run_job,
            args=(job, downloader_settings),
            name=f"job-{job.id}",
            daemon=True,
        ).start()
        return SubmitResponse(job_id=job.id, state=job.state)

    @app.post("/enrich", response_model=EnrichResponse)
    def enrich(
        file: UploadFile = File(...),
        url: str = Form(...),
    ) -> EnrichResponse:
        if _kind_from_url(url) != "track":
            raise HTTPException(
                status_code=400,
                detail="URL must be a Spotify track URL",
            )
        if not file.filename or not file.filename.lower().endswith(".mp3"):
            raise HTTPException(
                status_code=400,
                detail="Only .mp3 files are supported",
            )

        fd, tmp_path_str = tempfile.mkstemp(suffix=".mp3", prefix="enrich-")
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "wb") as out:
                shutil.copyfileobj(file.file, out)
            return _enrich_file(tmp_path, url, downloader_settings)
        finally:
            # If _enrich_file moved the temp file to its final location this
            # is a no-op; otherwise we leave nothing behind.
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    def status(job_id: str) -> JobStatus:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        with job.lock:
            return JobStatus(
                job_id=job.id,
                url=job.url,
                kind=job.kind,
                state=job.state,
                total=job.total,
                downloaded=job.downloaded,
                errored=job.errored,
                skipped=job.skipped,
                paths=list(job.paths),
                errored_tracks=list(job.errored_tracks),
                error=job.error,
            )

    host = server_settings.get("host") or "0.0.0.0"
    port = int(server_settings.get("port") or 8800)
    logger.info("spotdl api-server listening on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


def _run_job(job: Job, base_settings: DownloaderOptions) -> None:
    """Worker thread body for a single download job."""

    downloader: Optional[Downloader] = None
    try:
        with job.lock:
            job.kind = _kind_from_url(job.url)
            job.state = STATE_RESOLVING

        per_request: Dict[str, Any] = dict(base_settings)
        m3u_template = _m3u_template_for(job.url, base_settings["output"])
        if m3u_template:
            per_request["m3u"] = m3u_template
            Path(m3u_template.split("{", 1)[0]).expanduser().mkdir(
                parents=True, exist_ok=True
            )

        downloader = Downloader(per_request)

        songs = get_simple_songs(
            [job.url],
            use_ytm_data=downloader.settings["ytm_data"],
            playlist_numbering=downloader.settings["playlist_numbering"],
            albums_to_ignore=downloader.settings["ignore_albums"],
            album_type=downloader.settings["album_type"],
            playlist_retain_track_cover=downloader.settings[
                "playlist_retain_track_cover"
            ],
        )

        # `download_multiple_songs` archive-filters before doing any work; we
        # mirror that filter here so `total` reflects what will actually be
        # downloaded and the bot's progress bar advances at a sensible pace.
        archive_skipped = 0
        if downloader.settings.get("archive") and downloader.url_archive:
            archive_skipped = sum(
                1 for s in songs if s.url in downloader.url_archive
            )

        with job.lock:
            job.skipped = archive_skipped
            job.total = max(0, len(songs) - archive_skipped)
            job.state = STATE_DOWNLOADING

        # `search_and_download` is the per-song synchronous worker called
        # from each pool_download coroutine. Wrapping it gives us per-track
        # progress without reimplementing spotdl's orchestrator.
        original_search_and_download = downloader.search_and_download

        def tracked_search_and_download(song):
            result = original_search_and_download(song)
            with job.lock:
                if result[1] is not None:
                    job.downloaded += 1
                else:
                    job.errored += 1
                    job.errored_tracks.append(song.display_name)
            return result

        downloader.search_and_download = tracked_search_and_download  # type: ignore[assignment]

        results = downloader.download_multiple_songs(songs)

        with job.lock:
            job.paths = [str(p) for _s, p in results if p is not None]
            job.state = STATE_DONE
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("job %s failed", job.id)
        with job.lock:
            job.state = STATE_FAILED
            job.error = str(exc)
    finally:
        with job.lock:
            job.finished_at = time.time()
        if downloader is not None:
            try:
                downloader.progress_handler.close()
            except Exception:  # pylint: disable=broad-except
                pass


def _enrich_file(
    tmp_path: Path, url: str, base_settings: DownloaderOptions
) -> EnrichResponse:
    """
    Move an uploaded mp3 to its rightful place in the music library and embed
    Spotify-sourced tags + cover + lyrics. No re-encoding, no download — the
    user's audio is preserved bit-for-bit; only the ID3 frames change.
    """

    try:
        song = Song.from_url(url)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=400, detail=f"Failed to resolve Spotify URL: {exc}"
        ) from exc

    # Best-effort lyrics. Fetch from configured providers but never fail the
    # enrich if all of them error out — tags + cover already justify the call.
    song.lyrics = _fetch_lyrics(song, base_settings)

    output_file = create_file_name(
        song,
        base_settings["output"],
        base_settings["format"],
        base_settings.get("restrict"),
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(tmp_path), str(output_file))

    try:
        embed_metadata(
            output_file,
            song,
            id3_separator=base_settings.get("id3_separator", "/"),
            skip_album_art=bool(base_settings.get("skip_album_art", False)),
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=500, detail=f"Failed to embed metadata: {exc}"
        ) from exc

    # We just produced an audio file for this Spotify URL, so it belongs in
    # the archive — future playlist downloads containing this track will
    # skip the download leg. Failure here doesn't roll back the enrich; the
    # tagged file is already valuable on its own.
    _add_to_archive(song.url, base_settings)

    logger.info(
        "enriched %s → %s",
        song.display_name,
        output_file,
    )
    return EnrichResponse(
        url=url,
        output_path=str(output_file),
        song_name=song.name,
        song_artist=song.artist,
    )


def _add_to_archive(url: str, settings: DownloaderOptions) -> None:
    """
    Append a URL to the configured archive file. Re-reads the file under a
    process-wide lock so a concurrent download's save() can't drop our
    addition (and ours can't drop theirs).
    """

    archive_path = settings.get("archive")
    if not archive_path:
        return

    with _archive_lock:
        archive = Archive()
        archive.load(archive_path)  # silently no-op when file is absent
        if url in archive:
            return
        archive.add(url)
        try:
            archive.save(archive_path)
            logger.info("archive: added %s", url)
        except OSError as exc:
            logger.warning("archive: failed to save %s: %s", archive_path, exc)


def _fetch_lyrics(song: Song, settings: DownloaderOptions) -> Optional[str]:
    """
    Walk the configured lyrics-provider chain and return the first hit.
    Swallows provider errors so a single broken provider doesn't sink the
    whole enrich call.
    """

    for name in settings.get("lyrics_providers") or []:
        cls = LYRICS_PROVIDERS.get(name)
        if cls is None:
            continue
        try:
            provider = (
                cls(settings["genius_token"])
                if name == "genius" and settings.get("genius_token")
                else cls()
            )
            lyrics = provider.get_lyrics(song.name, list(song.artists))
            if lyrics:
                return lyrics
        except Exception:  # pylint: disable=broad-except
            logger.debug(
                "lyrics provider %s failed for %s",
                name,
                song.display_name,
                exc_info=True,
            )
    return None


def _apply_dedup_defaults(settings: DownloaderOptions) -> None:
    """
    Turn on the dedup mechanisms that make sense for a long-running bot, but
    only for keys the user hasn't explicitly set:

    - `archive`: O(1) skip-list of Spotify URLs we've already processed.
      Default location lives at the music root next to the audio files.
    - `scan_for_songs`: walks the output tree and reads each file's WOAS ID3
      tag at startup, catching duplicates whose filename or path changed.
    """

    if not settings.get("archive"):
        base = settings["output"].split("{", 1)[0].rstrip("/") or "."
        archive_path = str(Path(base).expanduser() / _ARCHIVE_FILENAME)
        Path(archive_path).parent.mkdir(parents=True, exist_ok=True)
        settings["archive"] = archive_path
        logger.info("dedup: archive enabled at %s", archive_path)
    else:
        logger.info("dedup: archive already set to %s", settings["archive"])

    if not settings.get("scan_for_songs"):
        settings["scan_for_songs"] = True
        logger.info("dedup: scan_for_songs enabled")
    else:
        logger.info("dedup: scan_for_songs already enabled by user")


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

    base = output_template.split("{", 1)[0].rstrip("/") or "."
    return f"{base}/Playlists/{{list[0]}}.m3u8"
