"""Recording → WAV → worker queue → transcription output."""
from __future__ import annotations

import wave
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.helpers import enqueue_and_wait, start_worker_and_wait, wav_duration

pytestmark = pytest.mark.unit


class TestTranscriptionPipeline:
    def test_transcribe_audio_writes_wav_and_enqueues(
        self, transcriber, tmp_path, monkeypatch
    ):
        vt = transcriber
        vt.should_save_debug_audio = True
        vt.debug_audio_dir = tmp_path / "debug_audio"
        vt.debug_audio_dir.mkdir()

        # Simulate one captured chunk (1024 int16 samples).
        chunk = (np.sin(np.linspace(0, 20, 1024)) * 5000).astype(np.int16).tobytes()
        vt.audio_data = [chunk]

        jobs = []

        class _CaptureQueue:
            def put(self, job):
                jobs.append(job)

        vt._transcription_queue = _CaptureQueue()
        vt.transcribe_audio()

        assert len(jobs) == 1
        wav_path = Path(jobs[0]["path"])
        assert wav_path.is_file()
        assert jobs[0]["audio_s"] > 0

        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1

        debug_copies = list(vt.debug_audio_dir.glob("recording_*.wav"))
        assert len(debug_copies) == 1

    def test_worker_mock_sidecar_full_pipeline(
        self, transcriber_with_mock_sidecar, audio_fixtures, capsys
    ):
        vt = transcriber_with_mock_sidecar
        start_worker_and_wait(vt, timeout=15)

        wav = audio_fixtures["tone_2s"]
        audio_s = wav_duration(wav)

        with patch("transcribe.pyperclip.copy") as mock_clip:
            enqueue_and_wait(vt, wav, audio_s, timeout=15)

        captured = capsys.readouterr().out
        assert "mock:tone_440hz_2s" in captured
        assert "Processed" in captured and f"{audio_s:.1f}s of audio" in captured
        mock_clip.assert_called_once_with("mock:tone_440hz_2s")
        vt._shutdown_asr()

    def test_process_transcription_no_speech_message(
        self, transcriber_with_mock_sidecar, audio_fixtures, capsys, monkeypatch
    ):
        vt = transcriber_with_mock_sidecar
        start_worker_and_wait(vt, timeout=15)

        def _empty(_path):
            return "", 0.05

        monkeypatch.setattr(vt, "_transcribe_via_sidecar", _empty)
        enqueue_and_wait(vt, audio_fixtures["silence_1s"], 1.0, timeout=15)

        captured = capsys.readouterr().out
        assert "No speech detected" in captured
        vt._shutdown_asr()

    def test_is_recording_gate(self, transcriber):
        vt = transcriber
        assert vt.is_recording is False
        vt.start_recording()
        assert vt.is_recording is True
        assert vt.audio_data == []
        vt.stop_recording()
        assert vt.is_recording is False
