#!/bin/bash
# Voice Transcriber Runner
# Activates virtual environment and runs the transcriber

cd "$(dirname "$0")"
source venv/bin/activate
python3 transcribe.py
