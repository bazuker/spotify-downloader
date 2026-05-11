# Pinned to 3.13 because `python:3-alpine` floats to the newest 3.x (3.14 at
# time of writing) and several wheels (notably rapidfuzz 3.12.x) aren't
# published for 3.14 yet, forcing a from-source build that fails on strict
# scikit-build-core pyproject parsing. 3.13 has full wheel coverage.
FROM python:3.13-alpine

# Install dependencies. `deno` is required by yt-dlp to evaluate YouTube's
# player JavaScript (signature/n-challenge decryption). yt-dlp's EJS
# subsystem auto-detects deno only — node works but needs explicit
# `--js-runtimes` config, which spotdl doesn't expose cleanly. Without a JS
# runtime, yt-dlp returns storyboard-only formats for modern YouTube videos
# and downloads fail with "Requested format is not available".
RUN apk add --no-cache \
    ca-certificates \
    ffmpeg \
    openssl \
    aria2 \
    g++ \
    git \
    deno \
    py3-cffi \
    libffi-dev \
    zlib-dev

# Install uv and update pip/wheel
RUN pip install --upgrade pip uv wheel spotipy

# Set workdir
WORKDIR /app

# Copy requirements files
COPY . .

# Install spotdl requirements
RUN uv sync

# Create a volume for the output directory
VOLUME /music

# Change Workdir to download location
WORKDIR /music

# Entrypoint command
ENTRYPOINT ["uv", "run", "--project", "/app", "spotdl"]
