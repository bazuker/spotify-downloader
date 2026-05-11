"""
Module that holds the entry point for the console.
"""

import cProfile
import logging
import os
import pstats
import signal
import sys
import time
from argparse import Namespace

from spotdl.console.api_server import api_server
from spotdl.console.download import download
from spotdl.console.meta import meta
from spotdl.console.save import save
from spotdl.console.sync import sync
from spotdl.console.url import url
from spotdl.console.web import web
from spotdl.download.downloader import Downloader, DownloaderError
from spotdl.utils.arguments import parse_arguments
from spotdl.utils.config import create_settings
from spotdl.utils.console import ACTIONS, generate_initial_config, is_executable
from spotdl.utils.downloader import check_ytmusic_connection
from spotdl.utils.ffmpeg import FFmpegError, download_ffmpeg, is_ffmpeg_installed
from spotdl.utils.logging import init_logging
from spotdl.utils.spotify import SpotifyClient, SpotifyError, save_spotify_cache

__all__ = ["console_entry_point", "OPERATIONS"]

OPERATIONS = {
    "download": download,
    "sync": sync,
    "save": save,
    "meta": meta,
    "url": url,
}

logger = logging.getLogger(__name__)

# Environment-variable overrides for credentials. Lets users avoid persisting
# secrets to ~/.spotdl/config.json — useful for Docker and CI runs where
# values come from the environment.
_ENV_OVERRIDES = (
    ("SPOTDL_CLIENT_ID", "client_id"),
    ("SPOTDL_CLIENT_SECRET", "client_secret"),
    ("SPOTDL_GENIUS_TOKEN", "genius_token"),
    ("SPOTDL_COOKIE_FILE", "cookie_file"),
)


def _apply_env_overrides(arguments: Namespace) -> None:
    """
    Copy SPOTDL_* environment values onto the parsed argument namespace, but
    only for flags the user did not pass on the command line. CLI flags still
    win; env vars beat the config file and built-in defaults.
    """

    overrode_creds = False
    for env_var, attr in _ENV_OVERRIDES:
        value = os.environ.get(env_var)
        if value and getattr(arguments, attr, None) is None:
            # cookie_file is special: yt-dlp errors on a missing path. Skip
            # silently when the file isn't there yet so the server still
            # starts cleanly for users who haven't dropped in cookies.txt.
            if attr == "cookie_file":
                # Print direct to stderr — this runs before init_logging() so
                # logger.info()/warning() may be swallowed by Python's default
                # log config, which would silently hide the cookie status.
                if not os.path.isfile(value):
                    print(
                        f"[spotdl] WARN: {env_var}={value} but the file does "
                        f"not exist; yt-dlp will run without cookies",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                print(
                    f"[spotdl] yt-dlp cookies loaded from {value} "
                    f"({os.path.getsize(value)} bytes)",
                    file=sys.stderr,
                    flush=True,
                )
            setattr(arguments, attr, value)
            if attr in ("client_id", "client_secret"):
                overrode_creds = True

    # Spotipy's CacheFileHandler keys tokens by file path, not by client_id.
    # If a previous run cached a token under the shared upstream creds, that
    # token would otherwise be replayed under our new env creds and we'd stay
    # rate-limited. Force in-memory caching whenever env vars supplied creds.
    if overrode_creds and getattr(arguments, "no_cache", None) is None:
        arguments.no_cache = True


def console_entry_point():
    """
    Entry point for the console. With profile flag, it runs the code with cProfile.
    """

    if "--profile" in sys.argv:
        with cProfile.Profile() as profile:
            entry_point()

        stats = pstats.Stats(profile)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats("spotdl.profile")
    else:
        entry_point()


def entry_point():
    """
    Console entry point for spotdl. This is where the magic happens.
    """

    # Create config file if it doesn't exist
    generate_initial_config()

    # Check if sys.argv contains an action
    # If it does, we run the action and exit
    try:
        action_to_run = next(
            action for action_name, action in ACTIONS.items() if action_name in sys.argv
        )
    except StopIteration:
        action_to_run = None

    if action_to_run:
        action_to_run()
        return None

    # Parse the arguments
    arguments = parse_arguments()
    _apply_env_overrides(arguments)

    # Create settings dicts
    spotify_settings, downloader_settings, web_settings = create_settings(arguments)

    init_logging(downloader_settings["log_level"], downloader_settings["log_format"])

    # If the application is frozen, we check for ffmpeg
    # if it's not present download it create config file
    if is_executable():
        if is_ffmpeg_installed() is False:
            download_ffmpeg()

    # Check if ffmpeg is installed
    if is_ffmpeg_installed(downloader_settings["ffmpeg"]) is False:
        raise FFmpegError(
            "FFmpeg is not installed. Please run `spotdl --download-ffmpeg` to install it, "
            "or `spotdl --ffmpeg /path/to/ffmpeg` to specify the path to ffmpeg."
        )

    # Check if we are not blocked by ytm
    if "youtube-music" in downloader_settings["audio_providers"]:
        if not check_ytmusic_connection():
            raise DownloaderError(
                "You are blocked by YouTube Music. "
                "Please use a VPN, change youtube-music to piped, or use other audio providers"
            )

    # Initialize spotify client
    SpotifyClient.init(**spotify_settings)
    spotify_client = SpotifyClient()

    # If the application is frozen start web ui
    # or if the operation is `web`
    if is_executable() or arguments.operation == "web":

        # Default to the current directory when running a frozen application
        if is_executable():
            web_settings["web_use_output_dir"] = True

        # Start web ui
        web(web_settings, downloader_settings)

        return None

    # Start the lightweight HTTP API (one POST /download endpoint). Reuses the
    # web operation's --host/--port flags via web_settings.
    if arguments.operation == "api-server":
        api_server(downloader_settings, web_settings)
        return None

    # Check if save file is present and if it's valid
    if isinstance(downloader_settings["save_file"], str) and (
        not downloader_settings["save_file"].endswith(".spotdl")
        and not downloader_settings["save_file"] == "-"
    ):
        raise DownloaderError("Save file has to end with .spotdl")

    # Check if the user is logged in
    if (
        arguments.query
        and "saved" in arguments.query
        and not spotify_settings["user_auth"]
    ):
        raise SpotifyError(
            "You must be logged in to use the saved query. "
            "Log in by adding the --user-auth flag"
        )

    # Initialize the downloader
    # for download, load and preload operations
    downloader = Downloader(downloader_settings)

    def graceful_exit(_signal, _frame):
        if spotify_settings["use_cache_file"]:
            save_spotify_cache(spotify_client.cache)

        downloader.progress_handler.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    start_time = time.perf_counter()

    try:
        # Pick the operation to perform
        # based on the name and run it!
        OPERATIONS[arguments.operation](
            query=arguments.query,
            downloader=downloader,
        )
    except Exception as exc:
        if downloader_settings["save_errors"]:
            with open(
                downloader_settings["save_errors"], "a", encoding="utf-8"
            ) as error_file:
                error_file.write("\n".join([exc + "\n" for exc in exc.args]))

            logger.debug("Saved errors to %s", downloader_settings["save_errors"])

        end_time = time.perf_counter()
        logger.debug("Took %d seconds", end_time - start_time)

        downloader.progress_handler.close()
        logger.exception("An error occurred")

        sys.exit(1)

    end_time = time.perf_counter()
    logger.debug("Took %d seconds", end_time - start_time)

    if spotify_settings["use_cache_file"]:
        save_spotify_cache(spotify_client.cache)

    downloader.progress_handler.close()

    return None
