#!/bin/bash
# Build the FluidAudio ASR sidecar (Swift). Prints the release binary path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/asr_sidecar"

if ! command -v swift &>/dev/null; then
    echo "❌ swift not found. Install Xcode 15+ or Xcode Command Line Tools." >&2
    exit 1
fi

echo "🔨 Building asr-sidecar (release)..." >&2
swift build -c release >&2

BIN="$SCRIPT_DIR/asr_sidecar/.build/release/asr-sidecar"
if [[ ! -x "$BIN" ]]; then
    echo "❌ Build finished but binary missing at $BIN" >&2
    exit 1
fi

echo "$BIN"
