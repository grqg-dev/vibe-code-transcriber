#!/bin/bash
# Run the test suite. Unit tests only by default; pass --all for MLX + FluidAudio.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d venv ]]; then
    echo "Run ./install.sh first" >&2
    exit 1
fi
source venv/bin/activate

pip install -q -r requirements-dev.txt

# Ensure committed audio fixtures exist.
if [[ ! -f tests/fixtures/audio/tone_440hz_2s.wav ]]; then
    python3 tests/fixtures/generate_audio_fixtures.py
fi

MODE="${1:---unit}"
shift || true
case "$MODE" in
    --unit)
        echo "🧪 Running unit tests..."
        pytest -m "not integration" --cov=transcribe --cov-report=term-missing "$@"
        ;;
    --all)
        echo "🧪 Running unit + integration tests..."
        pytest --cov=transcribe --cov-report=term-missing "$@"
        ;;
    *)
        pytest "$MODE" "$@"
        ;;
esac
