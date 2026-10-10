"""Tests for defensive audio validation: VAD, SNR, clipping, silence, duration."""

import numpy as np
import pytest

from voicepilot.audio.transcoder import transcode_to_pcm16_wav
from voicepilot.audio.validator import (
    ClippingError,
    NoSpeechError,
    TooLongError,
    TooNoisyError,
    TooShortError,
    validate_audio_samples,
)


def test_valid_speech_passes(speech_wav):
    _, samples, sr = transcode_to_pcm16_wav(speech_wav)
    metrics = validate_audio_samples(
        samples=samples,
        sample_rate=sr,
        min_speech_duration_seconds=0.5,
        max_audio_seconds=30.0,
        min_snr_db=10.0,
    )
    assert metrics.is_valid is True
    assert metrics.duration_seconds >= 1.5
    assert metrics.speech_duration_seconds >= 0.5
    assert metrics.snr_db >= 10.0
    assert metrics.clipping_ratio < 0.05


def test_silence_rejected_with_no_speech_error(silence_wav):
    _, samples, sr = transcode_to_pcm16_wav(silence_wav)
    with pytest.raises(NoSpeechError) as exc_info:
        validate_audio_samples(samples, sample_rate=sr)
    assert "no speech" in str(exc_info.value).lower()
    assert exc_info.value.code == "no_speech"


def test_noisy_audio_rejected_with_too_noisy_error(noisy_wav):
    _, samples, sr = transcode_to_pcm16_wav(noisy_wav)
    with pytest.raises((TooNoisyError, NoSpeechError)) as exc_info:
        validate_audio_samples(samples, sample_rate=sr, min_snr_db=10.0)
    assert exc_info.value.code in ("too_noisy", "no_speech")


def test_clipped_audio_rejected_with_clipping_error(clipped_wav):
    _, samples, sr = transcode_to_pcm16_wav(clipped_wav)
    with pytest.raises(ClippingError) as exc_info:
        validate_audio_samples(samples, sample_rate=sr, max_clipping_ratio=0.05)
    assert "clipping" in str(exc_info.value).lower()
    assert exc_info.value.code == "clipping"


def test_short_audio_rejected_with_too_short_error(short_wav):
    _, samples, sr = transcode_to_pcm16_wav(short_wav)
    with pytest.raises(TooShortError) as exc_info:
        validate_audio_samples(samples, sample_rate=sr)
    assert "too short" in str(exc_info.value).lower()
    assert exc_info.value.code == "too_short"


def test_oversized_audio_rejected_with_too_long_error():
    # 35 seconds of silence/tone
    sr = 16000
    long_samples = np.zeros(sr * 35, dtype=np.float32)
    with pytest.raises(TooLongError) as exc_info:
        validate_audio_samples(long_samples, sample_rate=sr, max_audio_seconds=30.0)
    assert "too long" in str(exc_info.value).lower()
    assert exc_info.value.code == "too_long"
