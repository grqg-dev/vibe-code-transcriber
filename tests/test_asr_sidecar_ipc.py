"""ASR sidecar IPC: mock subprocess and protocol edge cases."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.helpers import start_worker_and_wait, wav_duration

pytestmark = pytest.mark.unit


class TestMockSidecarProtocol:
    def test_mock_sidecar_ready_and_transcribe(self, mock_sidecar_path, audio_fixtures):
        proc = subprocess.Popen(
            [sys.executable, str(mock_sidecar_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            ready_line = proc.stdout.readline()
            ready = json.loads(ready_line)
            assert ready["type"] == "ready"

            wav = audio_fixtures["tone_2s"]
            req = {"type": "transcribe", "id": 1, "path": str(wav)}
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()

            result_line = proc.stdout.readline()
            result = json.loads(result_line)
            assert result["type"] == "result"
            assert result["ok"] is True
            assert result["text"] == "mock:tone_440hz_2s"
            assert result["audio_s"] == pytest.approx(wav_duration(wav), rel=0.01)

            proc.stdin.write('{"type":"quit"}\n')
            proc.stdin.flush()
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_transcribe_via_sidecar_end_to_end(
        self, transcriber_with_mock_sidecar, audio_fixtures, capsys
    ):
        vt = transcriber_with_mock_sidecar
        start_worker_and_wait(vt, timeout=15)

        wav = audio_fixtures["tone_2s"]
        text, processing_s = vt._transcribe_via_sidecar(str(wav))
        assert text == "mock:tone_440hz_2s"
        assert processing_s >= 0.0
        vt._shutdown_asr()

    def test_stdout_reader_fulfills_pending_request(self, transcriber):
        vt = transcriber
        event = threading.Event()
        box = {}
        vt._asr_pending[42] = (event, box)

        line = json.dumps(
            {
                "type": "result",
                "id": 42,
                "text": "hello",
                "processing_s": 0.5,
                "ok": True,
            }
        )
        # Simulate reader logic inline
        msg = json.loads(line)
        req_id = msg["id"]
        with vt._asr_lock:
            pending = vt._asr_pending.pop(req_id, None)
        pending[1]["msg"] = msg
        pending[0].set()

        assert event.is_set()
        assert box["msg"]["text"] == "hello"

    def test_broken_pipe_clears_sidecar_handle(self, transcriber, monkeypatch):
        vt = transcriber
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(0)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        vt._asr = proc
        proc.stdin.close()

        with pytest.raises(RuntimeError, match="pipe broken|not running"):
            vt._transcribe_via_sidecar("/tmp/does-not-matter.wav")
