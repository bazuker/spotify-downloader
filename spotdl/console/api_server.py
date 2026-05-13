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

import hashlib
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from spotdl.download.downloader import LYRICS_PROVIDERS, Downloader
from spotdl.providers.audio.base import AudioProvider
from spotdl.types.options import DownloaderOptions, WebOptions
from spotdl.types.song import Song
from spotdl.utils.archive import Archive
from spotdl.utils.formatter import create_file_name
from spotdl.utils.metadata import embed_metadata, get_file_metadata
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

# Subsonic API client identifier; surfaced in Navidrome's audit log.
_SUBSONIC_CLIENT = "musicdon"
_SUBSONIC_API_VERSION = "1.16.1"
_SUBSONIC_HTTP_TIMEOUT = 10  # seconds; Subsonic calls are tiny and synchronous
_NAVIDROME_SHARE_RE = re.compile(r"/share/([A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class NavidromeConfig:
    """
    Resolved Navidrome connection config, or None when disabled.

    Loaded once at api-server startup from NAVIDROME_URL / NAVIDROME_USER /
    NAVIDROME_PASSWORD. The Subsonic-compatible REST API at <url>/rest/* is
    used for share resolution.
    """

    base_url: str
    user: str
    password: str


def _navidrome_config() -> Optional[NavidromeConfig]:
    url = (os.environ.get("NAVIDROME_URL") or "").strip().rstrip("/")
    user = (os.environ.get("NAVIDROME_USER") or "").strip()
    pw = os.environ.get("NAVIDROME_PASSWORD") or ""
    if not url or not user or not pw:
        return None
    return NavidromeConfig(base_url=url, user=user, password=pw)


def _subsonic_auth_params(cfg: NavidromeConfig) -> Dict[str, str]:
    """
    Build the Subsonic token-auth query params. A fresh salt is generated
    per call so a captured request can't be replayed indefinitely.
    """

    salt = secrets.token_hex(8)
    token = hashlib.md5((cfg.password + salt).encode("utf-8")).hexdigest()
    return {
        "u": cfg.user,
        "t": token,
        "s": salt,
        "v": _SUBSONIC_API_VERSION,
        "c": _SUBSONIC_CLIENT,
        "f": "json",
    }


def _subsonic_get(cfg: NavidromeConfig, endpoint: str, params: Dict[str, str]) -> Dict[str, Any]:
    """
    Call a Subsonic .view endpoint and return the parsed `subsonic-response`
    body. Raises ValueError with the server-reported error on Subsonic-level
    failures so callers can surface a human message.
    """

    qs = {**_subsonic_auth_params(cfg), **params}
    url = f"{cfg.base_url}/rest/{endpoint}"
    resp = requests.get(url, params=qs, timeout=_SUBSONIC_HTTP_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    body = payload.get("subsonic-response") or {}
    if body.get("status") != "ok":
        err = body.get("error") or {}
        raise ValueError(err.get("message") or f"subsonic {endpoint} failed")
    return body


def _navidrome_share_id_from_url(url: str) -> Optional[str]:
    """Extract the share ID from a Navidrome share URL like .../share/<id>."""
    m = _NAVIDROME_SHARE_RE.search(url)
    return m.group(1) if m else None


def _navidrome_share_entries(cfg: NavidromeConfig, share_id: str) -> List[Dict[str, Any]]:
    """
    Resolve a share to its list of entries. The Subsonic API exposes only
    `getShares.view` (plural, all shares accessible to the authenticated
    user); we fetch the full list and filter by id client-side. The JSON
    quirk where single-element lists collapse to a dict is normalized here
    so callers always get a List[Dict].
    """

    body = _subsonic_get(cfg, "getShares.view", {})
    shares = (body.get("shares") or {}).get("share")
    if shares is None:
        raise ValueError(f"share not found: {share_id}")
    if isinstance(shares, dict):
        shares = [shares]
    match = next(
        (s for s in shares if isinstance(s, dict) and s.get("id") == share_id),
        None,
    )
    if match is None:
        raise ValueError(f"share not found: {share_id}")
    entries = match.get("entry")
    if entries is None:
        return []
    if isinstance(entries, dict):
        entries = [entries]
    return [e for e in entries if isinstance(e, dict) and not e.get("isDir")]


def _local_path_for_subsonic(
    subsonic_path: str, settings: DownloaderOptions
) -> Path:
    """
    Map a Subsonic `path` field to a filesystem path inside this container.
    Navidrome's container may have a different library mount than ours
    (Navidrome's `/music` could map to the same host dir as spotdl's
    `/music`, but the returned path could be absolute or relative, and the
    leading prefix may not match). We try several candidates and return the
    first that exists on disk; if none exist, the first candidate is
    returned so the caller can include it in the error.
    """

    for candidate in _local_path_candidates(subsonic_path, settings):
        if candidate.exists():
            return candidate
    # Nothing exists — return our best guess so the caller's error message
    # surfaces a real path.
    return next(iter(_local_path_candidates(subsonic_path, settings)))


def _local_path_candidates(
    subsonic_path: str, settings: DownloaderOptions
) -> List[Path]:
    """
    Candidate filesystem paths to try for a Subsonic-reported file. Order:
      1. path as-is (handles absolute paths that already match our view)
      2. library-root / path (handles relative paths)
      3. library-root / (path stripped of its leading anchor) — handles
         absolute paths that came from a container with a different prefix
      4. library-root / basename(path) — flat-layout fallback
    """

    base = settings["output"].split("{", 1)[0].rstrip("/") or "."
    base_path = Path(base).expanduser()
    sub = Path(subsonic_path)

    candidates: List[Path] = []
    if sub.is_absolute():
        candidates.append(sub)
        anchorless = sub.relative_to(sub.anchor)
        candidates.append(base_path / anchorless)
    else:
        candidates.append(base_path / sub)
    candidates.append(base_path / sub.name)

    seen: set = set()
    deduped: List[Path] = []
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        deduped.append(c)
    return deduped


def _read_woas(file_path: Path) -> Optional[str]:
    """
    Read the WOAS (Web Of Audio Source) frame from an MP3 file, which is
    where we stash the Spotify URL when a track was paired with Spotify.
    Returns None when the frame is absent or the file isn't an MP3.
    """

    if file_path.suffix.lower() != ".mp3":
        return None
    try:
        from mutagen.id3 import ID3  # noqa: PLC0415
        from mutagen.id3._util import ID3NoHeaderError  # noqa: PLC0415
    except Exception:  # pylint: disable=broad-except
        return None
    try:
        tags = ID3(str(file_path))
    except ID3NoHeaderError:
        return None
    except Exception:  # pylint: disable=broad-except
        return None
    frames = tags.getall("WOAS")
    if not frames:
        return None
    url = getattr(frames[0], "url", None) or ""
    return url.strip() or None


@dataclass
class NavidromeTrack:
    """One resolved entry from a Navidrome share."""

    path: str
    name: str
    artist: str
    spotify_url: Optional[str]


def _resolve_navidrome_share(
    share_url_or_id: str,
    cfg: NavidromeConfig,
    settings: DownloaderOptions,
) -> List[NavidromeTrack]:
    """
    Turn a Navidrome share URL (or bare share ID) into a list of local file
    paths annotated with the Spotify URL we recorded in WOAS at download
    time. `spotify_url` is None for files that were never paired with Spotify
    (e.g. a /skip'd manual upload).
    """

    share_id = (
        _navidrome_share_id_from_url(share_url_or_id) or share_url_or_id.strip()
    )
    if not share_id:
        raise ValueError("missing Navidrome share ID")
    entries = _navidrome_share_entries(cfg, share_id)
    tracks: List[NavidromeTrack] = []
    for entry in entries:
        sub_path = entry.get("path") or ""
        if not sub_path:
            continue
        local = _resolve_navidrome_entry_path(entry, sub_path, settings)
        tracks.append(
            NavidromeTrack(
                path=str(local),
                name=str(entry.get("title") or "").strip() or local.stem,
                artist=str(entry.get("artist") or "").strip() or "Unknown Artist",
                spotify_url=_read_woas(local),
            )
        )
    return tracks


def _resolve_navidrome_entry_path(
    entry: Dict[str, Any],
    sub_path: str,
    settings: DownloaderOptions,
) -> Path:
    """
    Find the on-disk path for a Subsonic entry. We try:
      1. Candidates derived from Navidrome's reported `path`.
      2. The canonical path spotdl would have written given the entry's
         metadata — this handles the common case where Navidrome
         synthesizes a metadata-shaped path (Artist/Album/Disc-Track -
         Title) that doesn't match spotdl's flat output template.

    Returns the first path that exists, or our best guess if nothing does.
    """

    candidates = _local_path_candidates(sub_path, settings)
    for c in candidates:
        if c.exists():
            return c

    canonical = _spotdl_canonical_path_for_entry(entry, settings)
    if canonical is not None and canonical.exists():
        return canonical

    # Surface the best guess so the error message points at something useful.
    if canonical is not None:
        return canonical
    return candidates[0]


def _spotdl_canonical_path_for_entry(
    entry: Dict[str, Any], settings: DownloaderOptions
) -> Optional[Path]:
    """
    Build the filesystem path spotdl would have written for this song,
    using the entry's `title` / `artist` / `albumArtist` / `album` /
    `track` / `discNumber` / `year` fields. Returns None if there isn't
    enough metadata to synthesize a sensible Song.
    """

    title = str(entry.get("title") or "").strip()
    if not title:
        return None

    # Navidrome's openSubsonic schema returns the album-artist as
    # `displayAlbumArtist` (string) or the first entry of `albumArtists`
    # (list of {id,name}). The legacy `albumArtist` field is often absent.
    # The plain `artist` field is the slash-joined display string for
    # multi-artist tracks ("SALUKI/Вышел покурить"), which is NOT what
    # spotdl uses for {album-artist} — that would synthesize a different
    # filename than what's on disk.
    album_artist = (
        str(entry.get("displayAlbumArtist") or "").strip()
        or str(entry.get("albumArtist") or "").strip()
    )
    if not album_artist:
        album_artists = entry.get("albumArtists")
        if isinstance(album_artists, list) and album_artists:
            first = album_artists[0]
            if isinstance(first, dict):
                album_artist = str(first.get("name") or "").strip()
    if not album_artist:
        album_artist = str(entry.get("artist") or "").strip()

    artist = str(entry.get("artist") or "").strip() or album_artist
    if not (artist or album_artist):
        return None
    album_artist = album_artist or artist
    album = str(entry.get("album") or title).strip() or title

    try:
        track_number = int(entry.get("track") or 1)
    except (TypeError, ValueError):
        track_number = 1
    try:
        disc_number = int(entry.get("discNumber") or 1)
    except (TypeError, ValueError):
        disc_number = 1
    try:
        year = int(entry.get("year") or 0)
    except (TypeError, ValueError):
        year = 0
    try:
        duration = int(entry.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0

    try:
        song = Song.from_missing_data(
            name=title,
            artists=[artist],
            artist=artist,
            genres=[],
            disc_number=disc_number,
            disc_count=1,
            album_name=album,
            album_artist=album_artist,
            album_id="",
            duration=duration,
            year=year,
            date="",
            track_number=track_number,
            tracks_count=1,
            song_id="",
            explicit=False,
            publisher="",
            url="",
            isrc=None,
            cover_url=None,
            copyright_text=None,
            download_url=None,
        )
        return create_file_name(
            song,
            settings["output"],
            settings["format"],
            settings.get("restrict"),
        )
    except Exception:  # pylint: disable=broad-except
        return None


class SubmitRequest(BaseModel):
    """Body of POST /download."""

    url: str


class EnrichResponse(BaseModel):
    """Returned by POST /enrich and POST /youtube."""

    url: str
    output_path: str
    song_name: str
    song_artist: str


class DeleteRequest(BaseModel):
    """Body of POST /delete. Exactly one of `url` or `share_id` must be set."""

    url: Optional[str] = None
    share_id: Optional[str] = None
    confirm: bool = False


class DeleteTarget(BaseModel):
    """One audio file scheduled for deletion (or just deleted)."""

    url: str
    path: str
    name: str


class DeleteResponse(BaseModel):
    """Returned by POST /delete."""

    kind: str
    targets: List[DeleteTarget]
    m3u_path: Optional[str] = None
    count: int
    confirmed: bool
    deleted_count: int = 0
    m3u_entries_stripped: int = 0
    errors: List[str] = []


class YouTubeRequest(BaseModel):
    """Body of POST /youtube.

    `spotify_url` is optional: when provided, audio comes from `youtube_url`
    but tags come from Spotify (and the Spotify URL is added to the archive).
    When omitted, tags are built from yt-dlp's own extracted fields.
    """

    youtube_url: str
    spotify_url: Optional[str] = None


class LookupRequest(BaseModel):
    """Body of POST /lookup."""

    urls: List[str]


class LookupResult(BaseModel):
    """One resolved Spotify track. `error` is set when the lookup failed and
    the other fields are empty. `in_library` reflects archive membership,
    which is our proxy for "we've processed this URL before"."""

    url: str
    name: Optional[str] = None
    artist: Optional[str] = None
    artists: List[str] = []
    in_library: bool = False
    error: Optional[str] = None


class LookupResponse(BaseModel):
    """Returned by POST /lookup."""

    results: List[LookupResult]


class ResolveNavidromeRequest(BaseModel):
    """Body of POST /resolve-navidrome."""

    share_url: str


class ResolveNavidromeTrack(BaseModel):
    """One resolved entry from a Navidrome share."""

    path: str
    name: str
    artist: str
    spotify_url: Optional[str] = None


class ResolveNavidromeResponse(BaseModel):
    """Returned by POST /resolve-navidrome."""

    share_id: str
    tracks: List[ResolveNavidromeTrack]


class AssociateRequest(BaseModel):
    """Body of POST /associate. The share must resolve to exactly one track."""

    share_url: str
    spotify_url: str


class AssociateResponse(BaseModel):
    """Returned by POST /associate."""

    path: str
    spotify_url: str
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
        url: str = Form(""),
    ) -> EnrichResponse:
        # Empty `url` means the user /skipped — tag with the file's own ID3
        # data + filename fallback, no Spotify lookup.
        if url and _kind_from_url(url) != "track":
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
            if url:
                return _enrich_file(tmp_path, url, downloader_settings)
            return _enrich_file_skip(tmp_path, file.filename, downloader_settings)
        finally:
            # If the enrich path moved the temp file to its final location
            # this is a no-op; otherwise we leave nothing behind.
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    @app.post("/youtube", response_model=EnrichResponse)
    def youtube(req: YouTubeRequest) -> EnrichResponse:
        if not req.youtube_url:
            raise HTTPException(status_code=400, detail="youtube_url is required")

        if req.spotify_url:
            if _kind_from_url(req.spotify_url) != "track":
                raise HTTPException(
                    status_code=400,
                    detail="spotify_url must be a Spotify track URL",
                )
            return _youtube_with_spotify(
                req.youtube_url, req.spotify_url, downloader_settings
            )
        return _youtube_skip_spotify(req.youtube_url, downloader_settings)

    @app.post("/delete", response_model=DeleteResponse)
    def delete(req: DeleteRequest) -> DeleteResponse:
        if bool(req.url) == bool(req.share_id):
            raise HTTPException(
                status_code=400,
                detail="exactly one of `url` or `share_id` is required",
            )
        if req.share_id:
            cfg = _navidrome_config()
            if cfg is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Navidrome is not configured "
                        "(NAVIDROME_URL/USER/PASSWORD)"
                    ),
                )
            return _run_delete_navidrome(
                req.share_id, req.confirm, downloader_settings, cfg
            )
        kind = _kind_from_url(req.url or "")
        if kind not in {"track", "album", "playlist"}:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported URL kind for /delete: {kind!r}",
            )
        return _run_delete(req.url or "", kind, req.confirm, downloader_settings)

    @app.post("/resolve-navidrome", response_model=ResolveNavidromeResponse)
    def resolve_navidrome(req: ResolveNavidromeRequest) -> ResolveNavidromeResponse:
        cfg = _navidrome_config()
        if cfg is None:
            raise HTTPException(
                status_code=400,
                detail="Navidrome is not configured (NAVIDROME_URL/USER/PASSWORD)",
            )
        share_id = _navidrome_share_id_from_url(req.share_url) or req.share_url.strip()
        if not share_id:
            raise HTTPException(
                status_code=400, detail="missing Navidrome share ID"
            )
        try:
            tracks = _resolve_navidrome_share(req.share_url, cfg, downloader_settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502, detail=f"Navidrome unreachable: {exc}"
            ) from exc
        return ResolveNavidromeResponse(
            share_id=share_id,
            tracks=[
                ResolveNavidromeTrack(
                    path=t.path,
                    name=t.name,
                    artist=t.artist,
                    spotify_url=t.spotify_url,
                )
                for t in tracks
            ],
        )

    @app.post("/associate", response_model=AssociateResponse)
    def associate(req: AssociateRequest) -> AssociateResponse:
        cfg = _navidrome_config()
        if cfg is None:
            raise HTTPException(
                status_code=400,
                detail="Navidrome is not configured (NAVIDROME_URL/USER/PASSWORD)",
            )
        if _kind_from_url(req.spotify_url) != "track":
            raise HTTPException(
                status_code=400,
                detail="spotify_url must be a Spotify track URL",
            )
        try:
            tracks = _resolve_navidrome_share(
                req.share_url, cfg, downloader_settings
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502, detail=f"Navidrome unreachable: {exc}"
            ) from exc
        if len(tracks) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"share resolves to {len(tracks)} tracks; "
                    "/associate requires a single-track share"
                ),
            )
        track = tracks[0]
        file_path = Path(track.path)
        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"file not found: {file_path}"
            )
        if file_path.suffix.lower() != ".mp3":
            raise HTTPException(
                status_code=400,
                detail=f"unsupported file format for /associate: {file_path.suffix}",
            )

        try:
            song = Song.from_url(req.spotify_url)
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(
                status_code=400,
                detail=f"failed to resolve Spotify URL: {exc}",
            ) from exc

        try:
            _write_woas(file_path, req.spotify_url)
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(
                status_code=500,
                detail=f"failed to write WOAS tag: {exc}",
            ) from exc

        _add_to_archive(req.spotify_url, downloader_settings)
        logger.info(
            "associated %s with %s", file_path, req.spotify_url
        )
        return AssociateResponse(
            path=str(file_path),
            spotify_url=req.spotify_url,
            song_name=song.name,
            song_artist=song.artist,
        )

    @app.post("/lookup", response_model=LookupResponse)
    def lookup(req: LookupRequest) -> LookupResponse:
        # Snapshot the archive once per call so we can answer in_library
        # without rereading the file for each URL. Acquiring the lock keeps
        # us coherent with a concurrent download that's mid-save.
        archive_path = downloader_settings.get("archive")
        archive_urls: set = set()
        if archive_path:
            with _archive_lock:
                arc = Archive()
                arc.load(archive_path)
                archive_urls = {u for u in arc}

        results: List[LookupResult] = []
        for url in req.urls:
            if _kind_from_url(url) != "track":
                results.append(LookupResult(url=url, error="not a track URL"))
                continue
            try:
                song = Song.from_url(url)
            except Exception as exc:  # pylint: disable=broad-except
                results.append(LookupResult(url=url, error=str(exc)))
                continue
            results.append(
                LookupResult(
                    url=url,
                    name=song.name,
                    artist=song.artist,
                    artists=list(song.artists),
                    in_library=url in archive_urls,
                )
            )
        return LookupResponse(results=results)

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

    # Remove any pre-existing files in the library carrying this Spotify URL
    # (other than output_file itself, which shutil.move will overwrite). This
    # makes "drop a song with the same Spotify URL" replace cleanly even when
    # the new computed filename differs from the existing one — e.g. the
    # previous file was /skip'd and tagged under a different artist/title.
    _cleanup_duplicates_for_url(song.url, output_file, base_settings)

    shutil.move(str(tmp_path), str(output_file))
    # tempfile.mkstemp creates files mode 0600 for security, and shutil.move
    # across filesystems (which /tmp → /music is, inside the container)
    # preserves those bits. Without this chmod the library file would be
    # owner-only and Navidrome — running as a non-root user — couldn't read
    # it, silently skipping the file during scans.
    os.chmod(output_file, 0o644)

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


def _enrich_file_skip(
    tmp_path: Path, original_filename: str, base_settings: DownloaderOptions
) -> EnrichResponse:
    """
    /enrich path when the user skipped Spotify association. Builds a Song
    from the file's existing ID3 tags (read via mutagen), filling missing
    fields by parsing the original filename for an "Artist - Title" pattern.
    Falls back to "Unknown" placeholders only when both sources are silent.

    The file's embedded cover art is preserved untouched — we pass
    `skip_album_art=True` to embed_metadata so it doesn't try to overwrite
    with a None cover_url.
    """

    file_meta = get_file_metadata(
        tmp_path, base_settings.get("id3_separator", "/")
    ) or {}

    filename_artist, filename_title = _parse_filename(original_filename)

    name = (file_meta.get("name") or filename_title or "Unknown Title").strip() or "Unknown Title"
    primary_artist = (
        file_meta.get("artist") or filename_artist or "Unknown Artist"
    ).strip() or "Unknown Artist"

    artists_raw = file_meta.get("artists")
    if isinstance(artists_raw, list) and artists_raw:
        artists = [str(a).strip() for a in artists_raw if str(a).strip()]
    else:
        artists = [primary_artist]

    album = (file_meta.get("album_name") or name).strip() or name
    album_artist = (file_meta.get("album_artist") or primary_artist).strip() or primary_artist

    song = Song.from_missing_data(
        name=name,
        artists=artists,
        artist=primary_artist,
        genres=list(file_meta.get("genres") or []),
        disc_number=int(file_meta.get("disc_number") or 1),
        disc_count=int(file_meta.get("disc_count") or 1),
        album_name=album,
        album_artist=album_artist,
        album_id="",
        duration=int(file_meta.get("duration") or 0),
        year=int(file_meta.get("year") or 0),
        date=str(file_meta.get("date") or ""),
        track_number=int(file_meta.get("track_number") or 1),
        tracks_count=int(file_meta.get("tracks_count") or 1),
        song_id="",
        explicit=False,
        publisher=str(file_meta.get("publisher") or ""),
        url=str(file_meta.get("url") or ""),
        isrc=file_meta.get("isrc"),
        cover_url=None,
        copyright_text=str(file_meta.get("copyright_text") or "") or None,
        download_url=None,
        lyrics=file_meta.get("lyrics"),
    )

    output_file = create_file_name(
        song,
        base_settings["output"],
        base_settings["format"],
        base_settings.get("restrict"),
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_path), str(output_file))
    # tempfile.mkstemp creates files mode 0600 for security, and shutil.move
    # across filesystems (which /tmp → /music is, inside the container)
    # preserves those bits. Without this chmod the library file would be
    # owner-only and Navidrome — running as a non-root user — couldn't read
    # it, silently skipping the file during scans.
    os.chmod(output_file, 0o644)

    try:
        embed_metadata(
            output_file,
            song,
            id3_separator=base_settings.get("id3_separator", "/"),
            skip_album_art=True,  # preserve the file's existing cover, if any
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=500, detail=f"Failed to embed metadata: {exc}"
        ) from exc

    logger.info(
        "enriched (skip) %s → %s",
        song.display_name,
        output_file,
    )
    return EnrichResponse(
        url="",
        output_path=str(output_file),
        song_name=song.name,
        song_artist=song.artist,
    )


def _parse_filename(filename: str) -> tuple:
    """
    Best-effort parse of a Telegram-supplied filename. Returns
    (artist, title) where each may be None.

    Recognises the de-facto convention `Artist - Title.mp3`; otherwise treats
    the whole stem as the title and leaves artist None for the caller to
    backfill from elsewhere.
    """

    stem = Path(filename).stem.strip()
    if not stem:
        return None, None
    parts = stem.split(" - ", 1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()
    return None, stem


def _youtube_with_spotify(
    youtube_url: str, spotify_url: str, base_settings: DownloaderOptions
) -> EnrichResponse:
    """
    Download audio from `youtube_url`, tag with Spotify metadata from
    `spotify_url`, and add the Spotify URL to the archive.
    """

    try:
        song = Song.from_url(spotify_url)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=400, detail=f"Failed to resolve Spotify URL: {exc}"
        ) from exc

    song.lyrics = _fetch_lyrics(song, base_settings)

    # Compute the path _run_single_song will write to, then sweep any other
    # files in the library already tagged with this Spotify URL. The
    # `overwrite: force` in _run_single_song already replaces a file sitting
    # at the canonical path; this catches the case where the previous copy
    # lived under a different name (e.g. /skip'd upload renamed via tags).
    canonical = create_file_name(
        song,
        base_settings["output"],
        base_settings["format"],
        base_settings.get("restrict"),
    )
    _cleanup_duplicates_for_url(spotify_url, canonical, base_settings)

    output_path, song = _run_single_song(song, youtube_url, base_settings)
    _add_to_archive(spotify_url, base_settings)

    return EnrichResponse(
        url=spotify_url,
        output_path=output_path,
        song_name=song.name,
        song_artist=song.artist,
    )


def _youtube_skip_spotify(
    youtube_url: str, base_settings: DownloaderOptions
) -> EnrichResponse:
    """
    Download audio from `youtube_url` and tag it with yt-dlp's extracted
    fields (track/artist/album/year if it's a YT Music URL, otherwise
    title/uploader). Cover art = the largest available video thumbnail.
    """

    provider = AudioProvider(
        output_format=base_settings["format"],
        cookie_file=base_settings.get("cookie_file"),
    )
    try:
        info = provider.get_download_metadata(youtube_url, download=False)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=400, detail=f"yt-dlp could not resolve URL: {exc}"
        ) from exc

    song = _song_from_youtube_info(info, youtube_url)
    output_path, song = _run_single_song(song, youtube_url, base_settings)

    return EnrichResponse(
        url=youtube_url,
        output_path=output_path,
        song_name=song.name,
        song_artist=song.artist,
    )


def _run_single_song(
    song: Song, youtube_url: str, base_settings: DownloaderOptions
) -> tuple:
    """
    Hand a pre-built Song to the regular Downloader, with `download_url` set
    to the YouTube URL so spotdl skips the search step and goes straight to
    yt-dlp. Returns (output_path, song). Raises HTTPException on failure.

    scan_for_songs / archive are disabled in the per-request settings: we
    only want one file processed, and the archive update (if any) is done by
    the caller after this returns successfully.
    """

    song.download_url = youtube_url

    per_request: Dict[str, Any] = dict(base_settings)
    per_request["scan_for_songs"] = False
    per_request["archive"] = None
    # Single-track flows (YouTube, enrich) have no playlist context. Clearing
    # m3u here suppresses the "M3U file name contains '{list}' but no lists
    # were provided" warning that fires when gen_m3u_files is invoked
    # without any songs carrying a list_name.
    per_request["m3u"] = None
    # The user explicitly asked for this one song — if a file already exists
    # at the target path (replace-list claim flow, re-pair after /skip,
    # etc.) replace it instead of silently skipping. spotdl's default
    # "skip" makes the bot report success while the file on disk never
    # changes.
    per_request["overwrite"] = "force"

    downloader = Downloader(per_request)
    try:
        results = downloader.download_multiple_songs([song])
    finally:
        try:
            downloader.progress_handler.close()
        except Exception:  # pylint: disable=broad-except
            pass

    if not results or results[0][1] is None:
        detail = downloader.errors[0] if downloader.errors else "Download failed"
        raise HTTPException(status_code=500, detail=detail)

    return str(results[0][1]), results[0][0]


def _song_from_youtube_info(info: Dict[str, Any], youtube_url: str) -> Song:
    """
    Build a minimal Song from yt-dlp's info_dict. Fills every field
    downstream code might dereference (notably the ones search_and_download
    checks at downloader.py:472-479 — None for any of them would trigger a
    reinit_song call that hits Spotify, which we explicitly don't want here).
    """

    track = (info.get("track") or info.get("title") or "Unknown Title").strip()

    artists_raw = info.get("artists")
    if isinstance(artists_raw, list) and artists_raw:
        artists = [str(a).strip() for a in artists_raw if str(a).strip()]
    else:
        artist_str = (
            info.get("artist") or info.get("uploader") or "Unknown Artist"
        )
        artists = [a.strip() for a in str(artist_str).split(",") if a.strip()]
    if not artists:
        artists = ["Unknown Artist"]

    album = (info.get("album") or track).strip()
    album_artist = str(info.get("album_artist") or artists[0]).strip()

    cover_url: Optional[str] = info.get("thumbnail")
    thumbs = info.get("thumbnails") or []
    if thumbs:
        best = max(
            (t for t in thumbs if isinstance(t, dict)),
            key=lambda t: (t.get("width") or 0) * (t.get("height") or 0),
            default=None,
        )
        if best and best.get("url"):
            cover_url = best["url"]

    year = 0
    if info.get("release_year"):
        try:
            year = int(info["release_year"])
        except (TypeError, ValueError):
            year = 0
    elif info.get("upload_date"):
        ud = str(info["upload_date"])
        if len(ud) >= 4 and ud[:4].isdigit():
            year = int(ud[:4])

    return Song.from_missing_data(
        name=track,
        artists=artists,
        artist=artists[0],
        genres=[],
        disc_number=1,
        disc_count=1,
        album_name=album,
        album_artist=album_artist,
        album_id="",
        duration=int(info.get("duration") or 0),
        year=year,
        date=str(info.get("upload_date") or ""),
        track_number=1,
        tracks_count=1,
        song_id=str(info.get("id") or youtube_url),
        explicit=False,
        publisher="",
        url=youtube_url,
        isrc=None,
        cover_url=cover_url,
        copyright_text=None,
        download_url=youtube_url,
    )


def _run_delete(
    url: str, kind: str, confirm: bool, settings: DownloaderOptions
) -> DeleteResponse:
    """
    Resolve the deletion scope, look up matching audio files via the WOAS
    ID3 tag, and (when confirm=True) actually remove them, strip their
    references from every m3u, and remove the playlist's own m3u for
    playlist-kind URLs. Archive entries for deleted files are removed too.
    """

    try:
        target_urls = _resolve_delete_scope(url, kind)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=400, detail=f"failed to resolve {url}: {exc}"
        ) from exc

    targets = _find_files_for_urls(target_urls, settings)
    m3u_path = (
        _m3u_path_for_playlist(url, settings) if kind == "playlist" else None
    )

    if not confirm:
        return DeleteResponse(
            kind=kind,
            targets=targets,
            m3u_path=m3u_path,
            count=len(targets),
            confirmed=False,
        )

    deleted_count = 0
    deleted_paths: List[str] = []
    deleted_urls: List[str] = []
    errors: List[str] = []
    for t in targets:
        try:
            Path(t.path).unlink()
            deleted_count += 1
            deleted_paths.append(t.path)
            deleted_urls.append(t.url)
        except FileNotFoundError:
            # File already gone — still treat as success and clean archive.
            deleted_urls.append(t.url)
        except OSError as exc:
            errors.append(f"unlink {t.path}: {exc}")

    if deleted_urls:
        _archive_remove_many(deleted_urls, settings)

    stripped = _strip_m3u_references(
        deleted_paths, settings, except_m3u=m3u_path
    )

    if m3u_path and Path(m3u_path).exists():
        try:
            Path(m3u_path).unlink()
        except OSError as exc:
            errors.append(f"unlink m3u {m3u_path}: {exc}")

    logger.info(
        "delete: kind=%s url=%s files=%d m3u_stripped=%d m3u_removed=%s",
        kind,
        url,
        deleted_count,
        stripped,
        bool(m3u_path),
    )
    return DeleteResponse(
        kind=kind,
        targets=targets,
        m3u_path=m3u_path,
        count=len(targets),
        confirmed=True,
        deleted_count=deleted_count,
        m3u_entries_stripped=stripped,
        errors=errors,
    )


def _cleanup_duplicates_for_url(
    spotify_url: str,
    keep_path: Optional[Path],
    settings: DownloaderOptions,
) -> int:
    """
    Delete any audio files in the library whose WOAS tag matches
    `spotify_url`, except for `keep_path` (which the caller is about to
    write to). Also strips those files' references from every m3u so
    playlists stay consistent. Returns the number of files removed.

    Called from /enrich and /youtube-with-spotify so that re-tagging or
    re-downloading a track that already has a copy somewhere in the
    library (under any path, not just the canonical one) actually
    replaces it — without this, a /skip'd upload tagged via /associate
    can leave the original duplicate sitting next to the new file.
    """

    targets = _find_files_for_urls([spotify_url], settings)
    if not targets:
        return 0

    keep_resolved = keep_path.resolve() if keep_path is not None else None
    deleted_paths: List[str] = []
    for t in targets:
        path = Path(t.path)
        try:
            if keep_resolved is not None and path.resolve() == keep_resolved:
                continue
        except OSError:
            pass
        try:
            path.unlink()
            deleted_paths.append(t.path)
        except FileNotFoundError:
            deleted_paths.append(t.path)
        except OSError as exc:
            logger.warning("could not remove duplicate %s: %s", path, exc)

    if deleted_paths:
        _strip_m3u_references(deleted_paths, settings)
        logger.info(
            "cleanup: removed %d duplicate file(s) for %s",
            len(deleted_paths),
            spotify_url,
        )
    return len(deleted_paths)


def _run_delete_navidrome(
    share_id: str,
    confirm: bool,
    settings: DownloaderOptions,
    cfg: NavidromeConfig,
) -> DeleteResponse:
    """
    Delete-by-Navidrome-share: resolve the share to file paths, build delete
    targets from those paths (no library WOAS scan needed), then on confirm
    unlink the files, drop their WOAS-derived URLs from the archive, and
    strip m3u references.
    """

    try:
        tracks = _resolve_navidrome_share(share_id, cfg, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=f"Navidrome unreachable: {exc}"
        ) from exc

    targets: List[DeleteTarget] = []
    for t in tracks:
        targets.append(
            DeleteTarget(
                url=t.spotify_url or "",
                path=t.path,
                name=Path(t.path).stem,
            )
        )

    if not confirm:
        return DeleteResponse(
            kind="navidrome",
            targets=targets,
            m3u_path=None,
            count=len(targets),
            confirmed=False,
        )

    deleted_count = 0
    deleted_paths: List[str] = []
    deleted_urls: List[str] = []
    errors: List[str] = []
    for t in targets:
        path = Path(t.path)
        logger.info("delete (navidrome): attempting unlink %s", path)
        try:
            path.unlink()
            deleted_count += 1
            deleted_paths.append(t.path)
            if t.url:
                deleted_urls.append(t.url)
        except FileNotFoundError:
            # Unlike the Spotify-URL /delete path (where targets came from a
            # WOAS scan of files that definitely exist), Navidrome's reported
            # `path` may not match our filesystem view. Surface that instead
            # of silently swallowing so the user sees what we tried.
            errors.append(f"not found: {t.path}")
        except OSError as exc:
            errors.append(f"unlink {t.path}: {exc}")

    if deleted_urls:
        _archive_remove_many(deleted_urls, settings)

    stripped = _strip_m3u_references(deleted_paths, settings)

    logger.info(
        "delete (navidrome): share=%s files=%d m3u_stripped=%d errors=%d",
        share_id,
        deleted_count,
        stripped,
        len(errors),
    )
    return DeleteResponse(
        kind="navidrome",
        targets=targets,
        m3u_path=None,
        count=len(targets),
        confirmed=True,
        deleted_count=deleted_count,
        m3u_entries_stripped=stripped,
        errors=errors,
    )


def _write_woas(file_path: Path, spotify_url: str) -> None:
    """
    Overwrite the WOAS frame on an MP3 with `spotify_url`. Other tags are
    preserved untouched. Used by /associate to retroactively bind a Spotify
    track to an existing file without disturbing the user's chosen metadata.
    """

    from mutagen.id3 import ID3, WOAS  # noqa: PLC0415
    from mutagen.id3._util import ID3NoHeaderError  # noqa: PLC0415

    try:
        tags = ID3(str(file_path))
    except ID3NoHeaderError:
        tags = ID3()
    tags.delall("WOAS")
    tags.add(WOAS(url=spotify_url))
    tags.save(str(file_path))


def _resolve_delete_scope(url: str, kind: str) -> List[str]:
    """Spotify track URLs in scope of the deletion. Hits Spotify API for
    album/playlist kinds to get the current member list."""

    if kind == "track":
        return [url]

    # Lazy import — only the album/playlist paths need the client.
    from spotdl.utils.spotify import SpotifyClient  # noqa: PLC0415

    client = SpotifyClient()
    urls: List[str] = []
    if kind == "playlist":
        page = client.playlist_items(url, additional_types=("track",))
        while page:
            for item in page.get("items") or []:
                track = item.get("track") if isinstance(item, dict) else None
                if track and track.get("external_urls", {}).get("spotify"):
                    urls.append(track["external_urls"]["spotify"])
            page = client.next(page) if page.get("next") else None
    elif kind == "album":
        page = client.album_tracks(url)
        while page:
            for track in page.get("items") or []:
                if track.get("external_urls", {}).get("spotify"):
                    urls.append(track["external_urls"]["spotify"])
            page = client.next(page) if page.get("next") else None
    return urls


def _find_files_for_urls(
    urls: List[str], settings: DownloaderOptions
) -> List[DeleteTarget]:
    """Walk the output tree once and pick files whose WOAS tag matches one
    of `urls`. Survives template/filename changes since the original
    download because the index key is the Spotify URL, not the file path."""

    if not urls:
        return []

    # Lazy import to avoid pulling search at module load time.
    from spotdl.utils.search import gather_known_songs  # noqa: PLC0415

    formats = settings.get("detect_formats") or [settings["format"]]
    index: Dict[str, List[Path]] = {}
    for fmt in formats:
        for u, paths in gather_known_songs(settings["output"], fmt).items():
            index.setdefault(u, []).extend(paths)

    wanted = set(urls)
    targets: List[DeleteTarget] = []
    for u, paths in index.items():
        if u not in wanted:
            continue
        for path in paths:
            targets.append(
                DeleteTarget(url=u, path=str(path), name=path.stem)
            )
    return targets


def _m3u_path_for_playlist(
    playlist_url: str, settings: DownloaderOptions
) -> Optional[str]:
    """Reconstruct the m3u path we would have written for this playlist by
    looking up its current name from Spotify and applying the same
    sanitization spotdl uses at write time."""

    from spotdl.utils.formatter import sanitize_string  # noqa: PLC0415
    from spotdl.utils.spotify import SpotifyClient  # noqa: PLC0415

    try:
        pl = SpotifyClient().playlist(playlist_url)
    except Exception:  # pylint: disable=broad-except
        return None
    if not pl or not pl.get("name"):
        return None

    base = settings["output"].split("{", 1)[0].rstrip("/") or "."
    sanitized_name = sanitize_string(pl["name"])
    if not sanitized_name:
        return None
    return str(Path(base).expanduser() / "Playlists" / f"{sanitized_name}.m3u8")


def _archive_remove_many(urls: List[str], settings: DownloaderOptions) -> int:
    """Drop the given URLs from the archive. Returns the count removed."""

    archive_path = settings.get("archive")
    if not archive_path or not urls:
        return 0
    with _archive_lock:
        archive = Archive()
        archive.load(archive_path)
        before = len(archive)
        for url in urls:
            archive.discard(url)
        removed = before - len(archive)
        if removed:
            try:
                archive.save(archive_path)
                logger.info("archive: removed %d url(s)", removed)
            except OSError as exc:
                logger.warning(
                    "archive: failed to rewrite %s: %s", archive_path, exc
                )
        return removed


def _strip_m3u_references(
    deleted_paths: List[str],
    settings: DownloaderOptions,
    except_m3u: Optional[str] = None,
) -> int:
    """
    Walk every m3u under `<music-root>/Playlists/` and remove lines that
    point at one of the just-deleted audio files. Also removes the
    immediately preceding `#EXTINF:` line for each removed entry so the m3u
    pairing stays consistent. Returns the total number of entries stripped.

    `except_m3u`: if set, this m3u is skipped — used when the caller is
    about to delete the file entirely (playlist-kind delete).
    """

    base = settings["output"].split("{", 1)[0].rstrip("/") or "."
    playlists_dir = Path(base).expanduser() / "Playlists"
    if not playlists_dir.exists():
        return 0

    paths_set = set(deleted_paths)
    total_stripped = 0

    for m3u in playlists_dir.glob("*.m3u8"):
        if except_m3u and str(m3u) == except_m3u:
            continue
        try:
            lines = m3u.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        new_lines: List[str] = []
        stripped_here = 0
        for line in lines:
            if line.strip() in paths_set:
                # Drop the immediately preceding #EXTINF: line if present.
                if new_lines and new_lines[-1].startswith("#EXTINF:"):
                    new_lines.pop()
                stripped_here += 1
                continue
            new_lines.append(line)

        if stripped_here:
            try:
                m3u.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                total_stripped += stripped_here
            except OSError as exc:
                logger.warning("failed to rewrite m3u %s: %s", m3u, exc)

    return total_stripped


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
