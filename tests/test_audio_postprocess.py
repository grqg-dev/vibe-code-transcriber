"""Audio post-processing pipeline tests."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestPostProcess:
    def test_empty_bytes_unchanged(self, transcriber):
        assert transcriber._post_process_audio(b"") == b""

    def test_silence_stays_near_silence(self, transcriber, audio_fixtures):
        raw = audio_fixtures["silence_1s"].read_bytes()
        # Skip WAV header — postprocess expects raw PCM frames only.
        pcm = raw[44:]
        out = transcriber._post_process_audio(pcm)
        samples = np.frombuffer(out, dtype=np.int16)
        assert samples.size == np.frombuffer(pcm, dtype=np.int16).size
        assert np.max(np.abs(samples)) < 500

    def test_tone_is_normalized(self, transcriber, audio_fixtures):
        pcm = audio_fixtures["tone_2s"].read_bytes()[44:]
        out = transcriber._post_process_audio(pcm)
        peak = np.max(np.abs(np.frombuffer(out, dtype=np.int16)))
        # Peak normalize targets ~-1 dBFS (~32767 * 10^(-1/20) ≈ 29204)
        assert peak > 20_000
        assert peak <= 32767

    def test_short_clip_does_not_crash(self, transcriber, audio_fixtures):
        pcm = audio_fixtures["short_100ms"].read_bytes()[44:]
        out = transcriber._post_process_audio(pcm)
        assert len(out) == len(pcm)

    def test_beep_notch_reduces_800hz_energy(self, transcriber, audio_fixtures):
        pcm = audio_fixtures["beep_speech"].read_bytes()[44:]
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        out = np.frombuffer(transcriber._post_process_audio(pcm), dtype=np.int16).astype(np.float32)

        # Compare energy in first 0.15s (beep region) — notch should attenuate.
        beep_len = int(0.15 * transcriber.RATE)
        in_rms = np.sqrt(np.mean(samples[:beep_len] ** 2))
        out_rms = np.sqrt(np.mean(out[:beep_len] ** 2))
        assert out_rms < in_rms * 0.85

    def test_disabled_postprocess_config(self, test_config, fake_pyaudio, audio_fixtures):
        import yaml
        from transcribe import VoiceTranscriber

        cfg = yaml.safe_load(test_config.read_text())
        cfg["audio_postprocess"] = False
        test_config.write_text(yaml.safe_dump(cfg))
        vt = VoiceTranscriber(config_path=str(test_config))
        assert vt.audio_postprocess is False
