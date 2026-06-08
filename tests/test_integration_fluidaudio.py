"""Integration tests: real Swift asr-sidecar + FluidAudio models."""
from __future__ import annotations

import pytest

from tests.helpers import enqueue_and_wait, start_worker_and_wait, wav_duration

pytestmark = [pytest.mark.integration, pytest.mark.fluidaudio]


@pytest.fixture
def fluid_transcriber(test_config, fake_pyaudio, asr_sidecar_binary):
    from transcribe import VoiceTranscriber

    vt = VoiceTranscriber(
        config_path=str(test_config),
        asr_backend_override="fluidaudio",
        asr_model_version_override="v2",
    )
    vt.asr_sidecar_path = asr_sidecar_binary
    vt.show_indicator = False
    vt.auto_paste = False
    vt.audio_feedback = False
    vt.attenuate_volume = False
    return vt


@pytest.mark.timeout(600)
def test_fluidaudio_transcribes_tone_fixture(fluid_transcriber, audio_fixtures, capsys):
    vt = fluid_transcriber
    start_worker_and_wait(vt, timeout=300)

    wav = audio_fixtures["tone_2s"]
    audio_s = wav_duration(wav)
    enqueue_and_wait(vt, wav, audio_s, timeout=300)

    out = capsys.readouterr().out
    assert "Processing transcription" in out
    assert f"{audio_s:.1f}s of audio" in out
    vt._shutdown_asr()


@pytest.mark.timeout(600)
def test_fluidaudio_sidecar_stays_warm_across_two_jobs(
    fluid_transcriber, audio_fixtures, capsys
):
    vt = fluid_transcriber
    start_worker_and_wait(vt, timeout=300)

    for key in ("tone_2s", "silence_1s"):
        wav = audio_fixtures[key]
        enqueue_and_wait(vt, wav, wav_duration(wav), timeout=120)

    out = capsys.readouterr().out
    assert out.count("Processing transcription") >= 2
    vt._shutdown_asr()
