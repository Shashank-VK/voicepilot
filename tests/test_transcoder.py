"""Tests for audio transcoding: WebM, MP4, WAV to 16kHz mono PCM."""

import pytest

from voicepilot.audio.transcoder import (
    AudioFormatError,
    check_ffmpeg_available,
    transcode_to_pcm16_wav,
)


def test_ffmpeg_available():
    assert check_ffmpeg_available() is True


def test_transcode_wav(speech_wav):
    wav_bytes, samples, sr = transcode_to_pcm16_wav(speech_wav)
    assert len(wav_bytes) > 44
    assert sr == 16000
    assert len(samples) > 0
    assert samples.dtype.name == "float32"
    assert -1.0 <= samples.min() and samples.max() <= 1.0


def test_transcode_webm_format(webm_audio):
    wav_bytes, samples, sr = transcode_to_pcm16_wav(webm_audio)
    assert len(wav_bytes) > 44
    assert sr == 16000
    assert len(samples) > 0


def test_transcode_mp4_format(mp4_audio):
    wav_bytes, samples, sr = transcode_to_pcm16_wav(mp4_audio)
    assert len(wav_bytes) > 44
    assert sr == 16000
    assert len(samples) > 0


def test_transcode_empty_bytes_raises_error():
    with pytest.raises(AudioFormatError) as exc_info:
        transcode_to_pcm16_wav(b"")
    assert "empty audio payload" in str(exc_info.value).lower()


def test_transcode_corrupt_data_raises_error():
    with pytest.raises(AudioFormatError) as exc_info:
        transcode_to_pcm16_wav(b"NOT_A_VALID_AUDIO_FILE_DATA_CORRUPT")
    assert "unsupported format" in str(exc_info.value).lower()
