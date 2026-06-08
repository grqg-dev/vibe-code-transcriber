"""Spectrum and waveform DSP for the indicator."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestDSP:
    @pytest.fixture
    def chunk(self):
        return np.sin(
            2 * np.pi * 440 * np.linspace(0, 1024 / 16000, 1024, endpoint=False)
        ).astype(np.int16) * 8000

    def test_spectrum_band_count_eq(self, transcriber, chunk):
        bands = transcriber._compute_spectrum_bands(chunk)
        assert len(bands) == 16
        assert all(0.0 <= b <= 1.0 for b in bands)

    def test_spectrum_louder_signal_higher_energy(self, transcriber):
        quiet = (np.sin(np.linspace(0, 50, 1024)) * 500).astype(np.int16)
        loud = (np.sin(np.linspace(0, 50, 1024)) * 12000).astype(np.int16)
        q_bands = transcriber._compute_spectrum_bands(quiet)
        l_bands = transcriber._compute_spectrum_bands(loud)
        assert max(l_bands) >= max(q_bands)

    def test_spectrogram_style_32_bands(self, test_config, fake_pyaudio):
        from transcribe import VoiceTranscriber

        vt = VoiceTranscriber(
            config_path=str(test_config),
            indicator_style_override="spectrogram",
        )
        chunk = np.zeros(1024, dtype=np.int16)
        bands = vt._compute_spectrum_bands(chunk)
        assert len(bands) == 32

    def test_waveform_orb_length(self, test_config, fake_pyaudio):
        from transcribe import VoiceTranscriber

        vt = VoiceTranscriber(
            config_path=str(test_config),
            indicator_style_override="orb",
        )
        chunk = (np.sin(np.linspace(0, 80, 1024)) * 10000).astype(np.int16)
        wf = vt._compute_waveform_samples(chunk)
        assert len(wf) == 128
        assert all(-1.0 <= x <= 1.0 for x in wf)

    def test_waveform_zero_chunk(self, test_config, fake_pyaudio):
        from transcribe import VoiceTranscriber

        vt = VoiceTranscriber(
            config_path=str(test_config),
            indicator_style_override="orb",
        )
        # Empty chunk → zero-filled waveform at orb resolution.
        assert vt._compute_waveform_samples(np.array([], dtype=np.int16)) == [0.0] * 128
