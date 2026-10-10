FROM python:3.12-slim

# Install system dependencies (FFmpeg required for WebM/MP4 audio transcoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests
COPY requirements.txt pyproject.toml ./

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and assets
COPY src/ ./src/
COPY docs/ ./docs/
COPY README.md LICENSE ./

# Install package in editable/wheel mode
RUN pip install --no-cache-dir -e .

# Expose HTTP & WebSocket port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start VoicePilot server
CMD ["python", "-m", "voicepilot.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
