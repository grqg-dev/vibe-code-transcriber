"""Configuration loading and backend validation."""
from __future__ import annotations

import pytest
import yaml
from pynput.keyboard import Key

pytestmark = pytest.mark.unit


class TestConfig:
    def test_default_config_has_required_keys(self, transcriber):
        cfg = transcriber.get_default_config()
        for key in (
            "asr_backend",
            "asr_model_version",
            "parakeet_model",
            "auto_paste",
            "audio",
        ):
            assert key in cfg

    def test_load_config_from_file(self, test_config, fake_pyaudio):
        from transcribe import VoiceTranscriber

        vt = VoiceTranscriber(config_path=str(test_config))
        assert vt.config["asr_backend"] == "fluidaudio"
        assert vt.config["audio"]["sample_rate"] == 16000

    def test_missing_config_falls_back_to_defaults(self, tmp_path, fake_pyaudio):
        from transcribe import VoiceTranscriber

        missing = tmp_path / "nope.yaml"
        vt = VoiceTranscriber(config_path=str(missing))
        assert vt.config["hotkey_code"] == "alt_r"

    def test_backend_override(self, test_config, fake_pyaudio):
        from transcribe import VoiceTranscriber

        vt = VoiceTranscriber(
            config_path=str(test_config),
            asr_backend_override="mlx",
            asr_model_version_override="v3",
        )
        assert vt.asr_backend == "mlx"
        assert vt.asr_model_version == "v3"

    def test_invalid_backend_falls_back_to_fluidaudio(self, test_config, fake_pyaudio, monkeypatch, capsys):
        from transcribe import VoiceTranscriber

        cfg = yaml.safe_load(test_config.read_text())
        cfg["asr_backend"] = "cloud"
        test_config.write_text(yaml.safe_dump(cfg))
        vt = VoiceTranscriber(config_path=str(test_config))
        assert vt.asr_backend == "fluidaudio"

    def test_hotkey_string_resolves_to_key(self, transcriber):
        assert transcriber.record_key == Key.alt_r

    def test_indicator_style_validation(self, test_config, fake_pyaudio, capsys):
        from transcribe import VoiceTranscriber

        vt = VoiceTranscriber(
            config_path=str(test_config),
            indicator_style_override="bogus",
        )
        assert vt.indicator_style == "eq"
        assert vt._spectrum_band_count == 16

    def test_orb_style_sets_waveform_length(self, test_config, fake_pyaudio):
        from transcribe import VoiceTranscriber

        vt = VoiceTranscriber(
            config_path=str(test_config),
            indicator_style_override="orb",
        )
        assert vt.indicator_style == "orb"
        assert vt._waveform_length == 128
        assert vt._spectrum_band_count == 0
