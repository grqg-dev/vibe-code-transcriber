#!/usr/bin/env python3
"""Drop-in mock for asr-sidecar: same JSON-line IPC, no CoreML."""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path


def _audio_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def main() -> None:
    print(json.dumps({"type": "ready"}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "quit":
            return
        if kind != "transcribe":
            continue
        req_id = msg.get("id")
        path = msg.get("path", "")
        stem = Path(path).stem if path else "unknown"
        text = f"mock:{stem}"
        print(
            json.dumps(
                {
                    "type": "result",
                    "id": req_id,
                    "text": text,
                    "audio_s": _audio_duration(path),
                    "processing_s": 0.001,
                    "ok": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
