"""Test helpers (importable from test modules; not pytest fixtures)."""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import wave
from pathlib import Path

FIXTURES_AUDIO = Path(__file__).resolve().parent / "fixtures" / "audio"


def start_worker_and_wait(vt, timeout=30.0):
    """Start _transcription_worker and block until _model_ready."""
    t = threading.Thread(target=vt._transcription_worker, daemon=True)
    t.start()
    if not vt._model_ready.wait(timeout=timeout):
        raise TimeoutError("ASR worker did not become ready")
    return t


def _job_wav_copy(wav_path: str | Path) -> Path:
    """Copy committed fixtures so the worker's delete-on-complete doesn't destroy them."""
    path = Path(wav_path).resolve()
    if FIXTURES_AUDIO in path.parents:
        dest = Path(tempfile.mkdtemp()) / path.name
        shutil.copy(path, dest)
        return dest
    return path


def enqueue_and_wait(vt, wav_path: str | Path, audio_s: float, timeout=30.0):
    """Put one job on the worker queue and wait for _process_transcription."""
    job_path = _job_wav_copy(wav_path)
    done = threading.Event()
    orig = vt._process_transcription

    def wrapped(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        finally:
            done.set()

    vt._process_transcription = wrapped
    vt._transcription_queue.put({
        "path": str(job_path),
        "audio_s": audio_s,
        "enqueued_at": time.perf_counter(),
    })
    if not done.wait(timeout=timeout):
        raise TimeoutError("transcription job timed out")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())
