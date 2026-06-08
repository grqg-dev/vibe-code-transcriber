#!/usr/bin/env python3
"""Regenerate committed test WAV fixtures under tests/fixtures/audio/."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "audio"
RATE = 16_000


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(clipped.tobytes())


def main() -> None:
    t1 = np.linspace(0, 1.0, RATE, endpoint=False)
    write_wav(OUT / "silence_1s.wav", np.zeros(RATE, dtype=np.float32))

    t2 = np.linspace(0, 2.0, RATE * 2, endpoint=False)
    tone = 0.25 * np.sin(2 * np.pi * 440 * t2) * 32767
    write_wav(OUT / "tone_440hz_2s.wav", tone)

    # Short clip — exercises filtfilt minimum-length guard in postprocess.
    short = 0.2 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.1, int(RATE * 0.1), endpoint=False)) * 32767
    write_wav(OUT / "short_100ms.wav", short)

    # Beep-like 800 Hz burst (mimics start-beep bleed for notch filter tests).
    beep_t = np.linspace(0, 0.15, int(RATE * 0.15), endpoint=False)
    beep = 0.35 * np.sin(2 * np.pi * 800 * beep_t) * 32767
    speech_t = np.linspace(0, 1.0, int(RATE * 1.0), endpoint=False)
    speech = 0.12 * np.sin(2 * np.pi * 180 * speech_t) * 32767
    mixed = np.concatenate([beep, speech])
    write_wav(OUT / "beep_then_speech_1s.wav", mixed)

    print(f"Wrote fixtures to {OUT}")
    for p in sorted(OUT.glob("*.wav")):
        print(f"  {p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
