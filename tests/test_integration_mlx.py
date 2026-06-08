"""Integration tests: real parakeet-mlx inference (slow, GPU)."""
from __future__ import annotations

import pytest

from tests.helpers import enqueue_and_wait, start_worker_and_wait, wav_duration

pytestmark = [pytest.mark.integration, pytest.mark.mlx]


@pytest.fixture
def mlx_transcriber(test_config, fake_pyaudio):
    from transcribe import VoiceTranscriber

    vt = VoiceTranscriber(
        config_path=str(test_config),
        asr_backend_override="mlx",
    )
    vt.show_indicator = False
    vt.auto_paste = False
    vt.audio_feedback = False
    vt.attenuate_volume = False
    return vt


@pytest.mark.timeout(300)
def test_mlx_transcribes_tone_fixture(mlx_transcriber, audio_fixtures, mlx_available, capsys):
    vt = mlx_transcriber
    start_worker_and_wait(vt, timeout=120)

    wav = audio_fixtures["tone_2s"]
    audio_s = wav_duration(wav)
    enqueue_and_wait(vt, wav, audio_s, timeout=120)

    out = capsys.readouterr().out
    assert "Processing transcription" in out
    # Pure tone fixture — often no speech; pipeline completing is the goal.
    vt._shutdown_asr()


@pytest.mark.timeout(300)
def test_mlx_worker_thread_owns_inference(mlx_transcriber, audio_fixtures, mlx_available):
    """Regression for MLX invariant #1: inference only on worker thread."""
    import threading

    vt = mlx_transcriber
    loaded = threading.Event()

    def _load_on_worker():
        vt.load_model()
        loaded.set()

    threading.Thread(target=_load_on_worker, daemon=True).start()
    assert loaded.wait(timeout=120)

    with pytest.raises(RuntimeError, match="Stream"):
        vt._transcribe_file(str(audio_fixtures["tone_2s"]))
