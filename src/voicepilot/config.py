"""Configuration settings for VoicePilot."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VoicePilot application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "VoicePilot"
    APP_VERSION: str = "0.1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Demo & Safety Controls
    DEMO_MODE: bool = True
    DEMO_API_KEY: str = "voicepilot-demo-key"

    # Provider Options: "groq" (cloud fast-path), "local" (faster-whisper), "mock" (CI/offline)
    STT_PROVIDER: Literal["groq", "local", "mock"] = "mock"
    TTS_PROVIDER: Literal["edge", "mock"] = "mock"
    LLM_PROVIDER: Literal["groq", "openai", "mock"] = "mock"

    # Cloud API Keys (optional if using mock or local)
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Audio Validation Thresholds
    MAX_AUDIO_SECONDS: float = 30.0
    MIN_SPEECH_DURATION_SECONDS: float = 0.5
    MIN_SNR_DB: float = 10.0
    MAX_SILENCE_RATIO: float = 0.85
    MAX_CLIPPING_RATIO: float = 0.05

    # Provider Timeouts & Network Safety
    PROVIDER_TIMEOUT_SECONDS: float = 8.0
    STT_MAX_RETRIES: int = 1

    # Local Whisper Configuration
    LOCAL_WHISPER_MODEL: str = "base.en"
    LOCAL_WHISPER_DEVICE: str = "cpu"
    LOCAL_WHISPER_COMPUTE_TYPE: str = "int8"

    # Edge TTS Configuration
    EDGE_TTS_VOICE: str = "en-US-JennyNeural"

    # Session Memory Limit
    MAX_SESSION_TURNS: int = 3

    # Logging
    LOG_LEVEL: str = "INFO"
    REDACT_LOGS: bool = True


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
