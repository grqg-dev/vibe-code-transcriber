#!/bin/bash
# Voice Transcriber Runner
# Activates virtual environment and runs the transcriber

set -euo pipefail
cd "$(dirname "$0")"

DEFAULT_SIDECAR="./asr_sidecar/.build/release/asr-sidecar"

# Auto-build the FluidAudio sidecar when using the default backend and binary is missing.
if [[ ! -x "$DEFAULT_SIDECAR" ]]; then
    BACKEND="${ASR_BACKEND:-}"
    if [[ -z "$BACKEND" ]] && [[ -f config.yaml ]]; then
        BACKEND="$(grep -E '^asr_backend:' config.yaml | head -1 | sed -E 's/^asr_backend:[[:space:]]*//; s/[[:space:]]+#.*//; s/[[:space:]]+$//; s/^["'\'']|["'\'']$//g')"
    fi
    if [[ -z "$BACKEND" || "$BACKEND" == "fluidaudio" ]]; then
        if command -v swift &>/dev/null; then
            echo "🔨 ASR sidecar not found — building (first time may take a few minutes)..."
            ./build_asr.sh >/dev/null
        else
            echo "⚠️  ASR sidecar missing and swift not found. Install Xcode 15+ or set asr_backend: mlx in config.yaml" >&2
            exit 1
        fi
    fi
fi

source venv/bin/activate
python3 transcribe.py "$@"
