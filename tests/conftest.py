"""Shared pytest fixtures for vibe-code-transcriber."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
AUDIO = FIXTURES / "audio"
MOCK_SIDECAR = FIXTURES / "mock_asr_sidecar.py"

sys.path.insert(0, str(ROOT))


class _FakePyAudio:
    """Minimal PyAudio stand-in so tests never touch CoreAudio."""

    paInt16 = 8

    def __init__(self):
        self._streams = []

    def open(self, **kwargs):
        stream = MagicMock()
        stream.read = MagicMock(return_value=b"\x00\x00" * kwargs.get("frames_per_buffer", 1024))
        stream.stop_stream = MagicMock()
        stream.close = MagicMock()
        self._streams.append(stream)
        return stream

    def get_sample_size(self, _fmt):
        return 2

    def terminate(self):
        pass


@pytest.fixture
def fake_pyaudio(monkeypatch):
    monkeypatch.setattr("transcribe.pyaudio.PyAudio", _FakePyAudio)


@pytest.fixture
def test_config(tmp_path) -> Path:
    """Minimal config.yaml for isolated tests."""
    cfg = {
        "asr_backend": "fluidaudio",
        "asr_model_version": "v2",
        "parakeet_model": "mlx-community/parakeet-tdt-0.6b-v3",
        "auto_paste": False,
        "audio_feedback": False,
        "attenuate_volume": False,
        "hotkey_code": "alt_r",
        "save_debug_audio": False,
        "show_indicator": False,
        "indicator_style": "eq",
        "indicator_anchor": "mouse",
        "audio_postprocess": True,
        "audio": {"sample_rate": 16000, "channels": 1, "chunk_size": 1024},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def transcriber(fake_pyaudio, test_config, monkeypatch):
    """VoiceTranscriber with hardware and side effects disabled."""
    from transcribe import VoiceTranscriber

    monkeypatch.setattr(
        "transcribe.VoiceTranscriber.get_system_volume",
        lambda self: 50,
    )
    monkeypatch.setattr(
        "transcribe.VoiceTranscriber.set_system_volume",
        lambda self, v: None,
    )
    vt = VoiceTranscriber(config_path=str(test_config))
    vt.show_indicator = False
    vt.should_save_debug_audio = False
    vt.auto_paste = False
    vt.audio_feedback = False
    vt.attenuate_volume = False
    return vt


@pytest.fixture
def audio_fixtures():
    """Paths to committed dummy WAV files."""
    paths = {
        "silence_1s": AUDIO / "silence_1s.wav",
        "tone_2s": AUDIO / "tone_440hz_2s.wav",
        "short_100ms": AUDIO / "short_100ms.wav",
        "beep_speech": AUDIO / "beep_then_speech_1s.wav",
    }
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        pytest.fail(
            f"Missing audio fixtures {missing}. "
            f"Run: python3 tests/fixtures/generate_audio_fixtures.py"
        )
    return paths


@pytest.fixture
def mock_sidecar_path():
    assert MOCK_SIDECAR.is_file(), f"Missing {MOCK_SIDECAR}"
    return MOCK_SIDECAR


@pytest.fixture
def transcriber_with_mock_sidecar(transcriber, mock_sidecar_path):
    """Point ASR at the Python mock sidecar instead of Swift binary."""
    transcriber.asr_backend = "fluidaudio"
    transcriber.asr_sidecar_path = mock_sidecar_path
    return transcriber


@pytest.fixture
def asr_sidecar_binary():
    """Real Swift sidecar path, or skip."""
    from transcribe import ASR_SIDECAR_DEFAULT

    if not ASR_SIDECAR_DEFAULT.is_file():
        pytest.skip("asr-sidecar binary not built (./build_asr.sh)")
    return ASR_SIDECAR_DEFAULT


@pytest.fixture
def mlx_available():
    try:
        import parakeet_mlx  # noqa: F401
    except ImportError:
        pytest.skip("parakeet-mlx not installed")
