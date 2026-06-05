#!/usr/bin/env python3
"""Non-interactive smoke: enqueue one WAV through VoiceTranscriber's worker path."""
import argparse
import sys
import tempfile
import wave
from pathlib import Path

# Repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transcribe import VoiceTranscriber  # noqa: E402


def make_test_wav(path: Path, seconds: float = 2.0, rate: int = 16000) -> None:
    import numpy as np

    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    # Quiet tone — enough energy that the model doesn't treat it as pure silence.
    samples = (0.15 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["fluidaudio", "mlx"], default="mlx")
    parser.add_argument("--wav", type=Path, help="Existing 16 kHz mono WAV")
    parser.add_argument("--model-version", choices=["v2", "v3"], default="v2")
    args = parser.parse_args()

    wav = args.wav
    if wav is None:
        wav = Path(tempfile.mkdtemp()) / "smoke.wav"
        make_test_wav(wav)
        print(f"Generated test WAV: {wav} ({wav.stat().st_size} bytes)")

    vt = VoiceTranscriber(
        asr_backend_override=args.backend,
        asr_model_version_override=args.model_version,
    )
    vt.show_indicator = False

    import threading

    worker = threading.Thread(target=vt._transcription_worker, daemon=True)
    worker.start()
    if not vt._model_ready.wait(timeout=900):
        print("❌ ASR never became ready", file=sys.stderr)
        sys.exit(1)
    if vt.asr_backend == "fluidaudio" and vt._asr is None:
        print("❌ FluidAudio sidecar failed to start", file=sys.stderr)
        sys.exit(1)

    print(f"✅ ASR ready ({args.backend})")

    import time
    import wave as wave_mod

    with wave_mod.open(str(wav), "rb") as wf:
        audio_s = wf.getnframes() / float(wf.getframerate())

    # Hand off through the worker queue (MLX must stay on that thread).
    done = threading.Event()
    orig_process = vt._process_transcription

    def _process_and_signal(*a, **kw):
        try:
            return orig_process(*a, **kw)
        finally:
            done.set()

    vt._process_transcription = _process_and_signal
    vt._transcription_queue.put({
        "path": str(wav),
        "audio_s": audio_s,
        "enqueued_at": time.perf_counter(),
    })
    if not done.wait(timeout=600):
        print("❌ Transcription timed out", file=sys.stderr)
        sys.exit(1)
    vt._shutdown_asr()


if __name__ == "__main__":
    main()
