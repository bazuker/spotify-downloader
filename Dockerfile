# Pinned to 3.13 because `python:3-alpine` floats to the newest 3.x (3.14 at
# time of writing) and several wheels (notably rapidfuzz 3.12.x) aren't
# published for 3.14 yet, forcing a from-source build that fails on strict
# scikit-build-core pyproject parsing. 3.13 has full wheel coverage.
FROM python:3.13-alpine

# Install dependencies
RUN apk add --no-cache \
    ca-certificates \
    ffmpeg \
    openssl \
    aria2 \
    g++ \
    git \
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
