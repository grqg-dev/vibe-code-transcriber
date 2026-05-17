#!/usr/bin/env python3
"""
Voice Transcriber - Push-to-talk terminal transcription tool
Hold dictation key to record, release to transcribe with NVIDIA Parakeet
running natively on Apple Silicon via the MLX framework (parakeet-mlx).
"""

import pyaudio
import wave
import pyperclip
import tempfile
import os
import sys
import json
from pynput import keyboard
from pynput.keyboard import Key, KeyCode, Controller
import threading
import time
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from queue import Queue
import numpy as np
import argparse
import atexit
import subprocess
import platform
import logging
import signal
import warnings

# Quiet the noisy stuff parakeet-mlx / huggingface_hub emit on import & load.
# These need to be set BEFORE the heavy import happens inside load_model().
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
# Keep HF download progress bars visible — they're useful on first run.

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Verbose debug logging — toggle with --verbose flag or VERBOSE=1 env var
VERBOSE = os.environ.get("VERBOSE", "0") == "1"


def dbg(msg):
    """Print a debug line if VERBOSE is on. Flushes immediately."""
    if VERBOSE:
        print(f"🐛 [debug] {msg}", flush=True)


class VoiceTranscriber:
    # Beep frequencies (Hz). The START beep plays AFTER is_recording flips
    # true, so it can leak through the speakers into the live mic; the
    # post-processing notch filter targets this frequency to suppress the
    # bleed. The STOP beep plays AFTER is_recording flips false, so it
    # never enters the recording and needs no filter.
    START_BEEP_HZ = 800
    STOP_BEEP_HZ = 600

    def __init__(self, config_path="config.yaml", real_time_mode=False,
                 indicator_style_override=None):
        # Load configuration
        self.config = self.load_config(config_path)

        self.is_recording = False
        self.audio_data = []
        self.audio = None
        self.record_start_time = None

        # Real-time mode settings
        self.real_time_mode = real_time_mode
        self.data_queue = Queue()
        self.phrase_time = None
        self.last_transcribed_text = ""
        self.transcription_thread = None

        # Initialize PyAudio with error handling
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            print(f"❌ Error initializing audio: {e}")
            print("Please check that your microphone is connected and accessible.")
            sys.exit(1)

        # Audio settings from config
        self.CHUNK = self.config['audio']['chunk_size']
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = self.config['audio']['channels']
        self.RATE = self.config['audio']['sample_rate']

        # Parakeet ASR model (will be loaded on first use)
        self.model = None
        self.model_name = self.config['parakeet_model']

        # Persistent input stream (opened in run())
        self.stream = None

        # Hotkey for recording from config.
        # Supports both numeric VK codes (e.g. 176) and pynput Key names (e.g. 'alt_r').
        hotkey_config = self.config['hotkey_code']
        if isinstance(hotkey_config, str):
            try:
                self.record_key = getattr(Key, hotkey_config)
            except AttributeError:
                print(f"❌ Unknown key name: {hotkey_config!r}", flush=True)
                print(f"   Use a numeric VK code or a valid pynput Key name "
                      f"(e.g. 'alt_r', 'shift', 'ctrl', 'cmd', 'f13').", flush=True)
                sys.exit(1)
        else:
            self.record_key = KeyCode.from_vk(hotkey_config)

        # Keyboard controller for auto-paste
        self.kb_controller = Controller()

        # Feature flags
        self.auto_paste = self.config['auto_paste']
        self.audio_feedback = self.config['audio_feedback']

        # Volume attenuation.
        # Capture the baseline ONCE at startup and always restore to it. The
        # previous "capture at attenuate, restore at stop" model raced badly:
        # on a fast tap, restore ran before the post-beep attenuate, leaving
        # the volume stuck low forever. With a fixed baseline + lock, restore
        # is always meaningful and idempotent.
        self.attenuate_volume = self.config.get('attenuate_volume', True)
        self.attenuation_percent = self.config.get('attenuation_percent', 10)
        self._volume_lock = threading.Lock()
        self._baseline_volume = self.get_system_volume() if self.attenuate_volume else None
        if self._baseline_volume is not None:
            dbg(f"baseline system volume captured at startup: {self._baseline_volume}%")
            # Last-ditch safety: if the program dies in any way that doesn't
            # go through our cleanup paths, atexit still fires and the user's
            # speakers don't stay stuck at 10%.
            atexit.register(self.restore_system_volume)

        # Post-processing on the recorded audio before it hits the model:
        # high-pass at 80 Hz (DC + rumble), notch at START_BEEP_HZ to
        # suppress the start-beep bleed, and peak-normalize to -1 dBFS.
        # Cheap (scipy filtfilt on a few seconds of 16 kHz audio is sub-ms)
        # and strictly improves WER on quiet speakers / noisy rooms.
        self.audio_postprocess = self.config.get('audio_postprocess', True)

        # Debug audio dump (helps verify mic actually captured something)
        self.should_save_debug_audio = self.config.get('save_debug_audio', True)
        self.debug_audio_dir = Path(__file__).parent / "debug_audio"
        if self.should_save_debug_audio:
            self.debug_audio_dir.mkdir(exist_ok=True)

        # MLX GPU streams are per-thread, so all model load/inference work has
        # to happen on a single dedicated worker thread. The pynput listener
        # callback runs on its own thread and cannot touch the model directly.
        self._transcription_queue = Queue()
        self._worker_thread = None
        self._model_ready = threading.Event()

        # Persistent mic capture: open once at startup, run forever, gate
        # append-to-buffer with is_recording. This makes the beep a true
        # "now recording" cue with zero key-press → audio-on latency, and
        # avoids the CoreAudio reconfigure that an on-press open caused.
        self._capture_thread = None
        self._capture_running = False

        # Optional floating visualization (sidecar process; see indicator.py).
        # Three styles supported; payload shape varies per style and is
        # locked to the indicator's STYLES table — keep these in sync.
        # Failure to spawn the indicator MUST be soft — the recorder
        # works fine without it.
        self.show_indicator = self.config.get('show_indicator', True)
        # CLI --style overrides config; both fall back to 'eq'.
        self.indicator_style = indicator_style_override or self.config.get('indicator_style', 'eq')
        if self.indicator_style not in ('eq', 'spectrogram', 'orb'):
            print(f"⚠️  Unknown indicator_style {self.indicator_style!r}, "
                  f"falling back to 'eq'", flush=True)
            self.indicator_style = 'eq'
        if indicator_style_override:
            dbg(f"indicator_style overridden via CLI: {self.indicator_style}")
        # Per-style payload sizes. MUST match the corresponding constants
        # in indicator.py (EQ_BAND_COUNT, SG_ROWS, ORB_WAVEFORM_LENGTH).
        self._spectrum_band_count = {
            'eq': 16, 'spectrogram': 32, 'orb': 0,
        }[self.indicator_style]
        self._waveform_length = 128 if self.indicator_style == 'orb' else 0
        self._indicator = None

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            config_file = Path(__file__).parent / config_path
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"⚠️  Config file not found at {config_path}, using defaults")
            return self.get_default_config()
        except Exception as e:
            print(f"⚠️  Error loading config: {e}, using defaults")
            return self.get_default_config()

    def get_default_config(self):
        """Return default configuration"""
        return {
            'parakeet_model': 'mlx-community/parakeet-tdt-0.6b-v3',
            'auto_paste': True,
            'audio_feedback': True,
            'hotkey_code': 'alt_r',
            'save_debug_audio': True,
            'show_indicator': True,
            'indicator_style': 'eq',
            'audio_postprocess': True,
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'chunk_size': 1024
            }
        }

    def play_beep(self, frequency=800, duration=0.1):
        """Play a beep sound for audio feedback.

        Uses the shared `self.audio` PyAudio instance — creating a fresh
        `pyaudio.PyAudio()` here causes CoreAudio on macOS to reconfigure
        the audio session and visibly glitch the live input stream.
        """
        if not self.audio_feedback:
            dbg(f"play_beep({frequency}Hz) skipped — audio_feedback disabled")
            return

        dbg(f"play_beep({frequency}Hz, {duration}s) starting")
        try:
            sample_rate = 44100
            samples = int(sample_rate * duration)
            t = np.linspace(0, duration, samples, False)
            wave_data = np.sin(frequency * 2 * np.pi * t)
            audio_data = (wave_data * 3000).astype(np.int16)

            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
            )
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
            dbg(f"play_beep({frequency}Hz) finished")
        except Exception as e:
            print(f"⚠️  Beep failed ({frequency}Hz): {type(e).__name__}: {e}", flush=True)

    def get_system_volume(self):
        """Get current system volume (macOS only)"""
        if platform.system() != 'Darwin':
            return None

        try:
            result = subprocess.run(
                ['osascript', '-e', 'output volume of (get volume settings)'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return None

    def set_system_volume(self, volume):
        """Set system volume (macOS only, 0-100)"""
        if platform.system() != 'Darwin':
            return False

        try:
            volume = max(0, min(100, int(volume)))  # Clamp to 0-100
            subprocess.run(
                ['osascript', '-e', f'set volume output volume {volume}'],
                capture_output=True,
                timeout=1
            )
            return True
        except Exception:
            return False

    def attenuate_system_volume(self):
        """Lower system volume to attenuation_percent of the baseline.

        Bails out if we're no longer recording, so a fast tap-and-release
        followed by a post-beep attenuate doesn't strand the volume low.
        """
        if not self.attenuate_volume or self._baseline_volume is None:
            return
        with self._volume_lock:
            if not self.is_recording:
                dbg("attenuate skipped — no longer recording (race avoided)")
                return
            target_volume = int(self._baseline_volume * (self.attenuation_percent / 100.0))
            self.set_system_volume(target_volume)
            dbg(f"attenuated volume: {self._baseline_volume}% → {target_volume}%")

    def restore_system_volume(self):
        """Restore system volume to the program-start baseline. Idempotent."""
        if not self.attenuate_volume or self._baseline_volume is None:
            return
        with self._volume_lock:
            self.set_system_volume(self._baseline_volume)
            dbg(f"restored volume to baseline: {self._baseline_volume}%")

    def check_microphone_access(self):
        """Check if microphone is accessible"""
        try:
            test_stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            test_stream.close()
            return True
        except Exception as e:
            return False

    def load_model(self):
        """Load Parakeet ASR model via MLX (downloads on first run)."""
        if self.model is not None:
            dbg("load_model: model already loaded, skipping")
            return

        print(f"⏳ Loading Parakeet (MLX) model '{self.model_name}'...", flush=True)
        print("   (First run downloads ~600MB — may take a minute)", flush=True)

        t0 = time.time()
        try:
            print("   → importing parakeet_mlx ...", flush=True)
            t_import = time.time()
            from parakeet_mlx import from_pretrained
            dbg(f"parakeet_mlx import took {time.time() - t_import:.1f}s")

            print("   → fetching / loading model weights ...", flush=True)
            t_load = time.time()
            self.model = from_pretrained(self.model_name)
            dbg(f"from_pretrained took {time.time() - t_load:.1f}s")

            print(f"✅ Model loaded successfully in {time.time() - t0:.1f}s\n", flush=True)
        except Exception as e:
            print(f"❌ Error loading Parakeet model: {type(e).__name__}: {e}", flush=True)
            print("   Check your internet connection and that parakeet-mlx is installed.", flush=True)
            print("   pip install: pip install parakeet-mlx -U", flush=True)
            if VERBOSE:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    def warmup_model(self):
        """Run one short inference so the first real transcribe isn't slow."""
        if self.model is None:
            return
        try:
            print("🔥 Warming up model (1s of silence)...", flush=True)
            t0 = time.time()
            silence = np.zeros(self.RATE, dtype=np.int16)  # 1s of silence at sample rate
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            wf = wave.open(tmp_path, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            wf.writeframes(silence.tobytes())
            wf.close()
            try:
                self._transcribe_file(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(f"✅ Model warm ({time.time() - t0:.1f}s)\n", flush=True)
        except Exception as e:
            print(f"⚠️  Warmup failed (non-fatal): {type(e).__name__}: {e}", flush=True)

    def _transcribe_file(self, wav_path):
        """Run Parakeet (MLX) inference on a WAV file and return the text."""
        result = self.model.transcribe(wav_path)
        # parakeet-mlx returns an AlignedResult with a `.text` attribute.
        # Fall back to str() in case the API ever yields a bare string.
        if hasattr(result, "text"):
            return (result.text or "").strip()
        return str(result).strip()

    def start_recording(self):
        """Begin recording. Mic stream is already live — just flip the flag."""
        dbg("start_recording called")
        if self.is_recording:
            dbg("start_recording: already recording, ignoring")
            return

        # Reset buffers BEFORE flipping the flag so the capture loop can't
        # append onto a stale audio_data list.
        self.audio_data = []
        self.record_start_time = time.time()
        self.last_transcribed_text = ""
        if self.real_time_mode:
            # Drain any stale queued chunks from the previous session.
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except Exception:
                    break
        self.is_recording = True

        if self.real_time_mode:
            self.transcription_thread = threading.Thread(
                target=self._real_time_transcribe, daemon=True
            )
            self.transcription_thread.start()
            dbg("real-time transcription thread started")

        # Pop the floating spectrum at the text-caret position (preferred)
        # or the mouse cursor (fallback) BEFORE the beep so the visual lines
        # up with the audio cue. _get_indicator_anchor returns AppKit-coord
        # top-left, exactly what the indicator expects.
        if self._indicator is not None:
            anchor = self._get_indicator_anchor()
            if anchor is not None:
                self._send_to_indicator(
                    {"type": "show", "x": float(anchor[0]), "y": float(anchor[1])}
                )

        if self.real_time_mode:
            print("\n" + "─" * 60)
            print("🔴 REAL-TIME RECORDING... (release key to stop)")
            print("─" * 60)
        else:
            print("\n" + "─" * 60)
            print("🔴 RECORDING... (release key to stop)")
            print("─" * 60)
        sys.stdout.flush()

        # Beep + attenuate run async so they don't delay anything. The mic is
        # ALREADY hot, so the beep is a truthful "now recording" cue.
        def beep_then_attenuate():
            dbg("beep_then_attenuate thread started")
            self.play_beep(self.START_BEEP_HZ, 0.1)
            self.attenuate_system_volume()
            dbg("beep_then_attenuate thread done")

        threading.Thread(target=beep_then_attenuate, daemon=True).start()

    def _capture_loop(self):
        """Persistent mic reader.

        Runs from `run()` until the process exits. Always drains the input
        stream so the OS buffer never backs up, but only appends to
        `self.audio_data` when `is_recording` is True. This means key-press
        latency is essentially zero — we don't pay for opening the stream
        on press, and there's no glitch from CoreAudio reconfiguring.
        """
        dbg("_capture_loop entered")
        chunks_appended = 0
        while self._capture_running:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            except Exception as e:
                if self._capture_running:
                    print(f"❌ Error during capture: {type(e).__name__}: {e}", flush=True)
                    if VERBOSE:
                        import traceback
                        traceback.print_exc()
                break

            if not self.is_recording:
                continue

            # Drive the floating indicator. Branch by style: bar/waterfall
            # styles want a frequency spectrum, orb wants a time-domain
            # waveform. Both pipelines run in tens of microseconds per
            # chunk on M-series — no measurable impact on capture latency.
            if self._indicator is not None:
                try:
                    samples = np.frombuffer(data, dtype=np.int16)
                    if samples.size:
                        if self.indicator_style == 'orb':
                            wave = self._compute_waveform_samples(samples)
                            self._send_to_indicator(
                                {"type": "waveform", "samples": wave}
                            )
                        else:
                            bands = self._compute_spectrum_bands(samples)
                            self._send_to_indicator(
                                {"type": "spectrum", "bands": bands}
                            )
                except Exception as e:
                    dbg(f"indicator send failed: {type(e).__name__}: {e}")

            if self.real_time_mode:
                self.data_queue.put(data)
            else:
                self.audio_data.append(data)
                chunks_appended += 1
        dbg(f"_capture_loop exited ({chunks_appended} chunks captured this session)")

    def _real_time_transcribe(self):
        """Real-time transcription thread - processes audio as it comes in"""
        # Load model if not already loaded
        self.load_model()

        phrase_bytes = bytes()
        record_timeout = 1.5  # How often to transcribe (in seconds)
        min_audio_length = 0.5  # Minimum audio length to transcribe (in seconds)

        print("💡 Starting real-time transcription...")
        sys.stdout.flush()

        while self.is_recording or not self.data_queue.empty():
            try:
                now = datetime.utcnow()

                # Check if we have audio data in the queue
                if not self.data_queue.empty():
                    # Update phrase time
                    if self.phrase_time is None:
                        self.phrase_time = now

                    # Get all available audio from queue
                    audio_chunks = []
                    while not self.data_queue.empty():
                        audio_chunks.append(self.data_queue.get())

                    # Combine and add to accumulated audio
                    audio_data = b''.join(audio_chunks)
                    phrase_bytes += audio_data

                    # Check if enough time has passed to transcribe
                    time_since_phrase_start = (now - self.phrase_time).total_seconds()
                    audio_duration = len(phrase_bytes) / (2 * self.CHANNELS * self.RATE)  # 2 bytes per sample

                    if audio_duration >= min_audio_length and time_since_phrase_start >= record_timeout:
                        # parakeet-mlx transcribes from a file path, so dump the
                        # accumulated phrase to a temporary WAV each pass.
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                tmp_path = tmp.name
                                wf = wave.open(tmp_path, 'wb')
                                wf.setnchannels(self.CHANNELS)
                                wf.setsampwidth(2)  # paInt16 = 2 bytes
                                wf.setframerate(self.RATE)
                                wf.writeframes(phrase_bytes)
                                wf.close()

                            try:
                                new_text = self._transcribe_file(tmp_path)
                            finally:
                                try:
                                    os.remove(tmp_path)
                                except OSError:
                                    pass

                            # Type only the new characters
                            if new_text and new_text != self.last_transcribed_text:
                                self._type_new_text(new_text)
                                self.last_transcribed_text = new_text

                        except Exception as e:
                            print(f"\n⚠️  Transcription error: {e}")

                        # Reset for next phrase
                        self.phrase_time = now
                        phrase_bytes = bytes()

                else:
                    # No data available, sleep briefly
                    time.sleep(0.1)

            except Exception as e:
                print(f"\n❌ Real-time transcription error: {e}")
                import traceback
                traceback.print_exc()
                break

        print("\n✅ Real-time transcription completed")
        sys.stdout.flush()

    def _type_new_text(self, new_text):
        """Type only the new portion of text to avoid duplicates"""
        # Find the common prefix between old and new text
        common_length = 0
        for i in range(min(len(self.last_transcribed_text), len(new_text))):
            if self.last_transcribed_text[i] == new_text[i]:
                common_length += 1
            else:
                break

        # Extract only the new characters
        new_chars = new_text[common_length:]

        if new_chars:
            # Type character by character
            for char in new_chars:
                try:
                    self.kb_controller.type(char)
                    time.sleep(0.01)  # Small delay between characters for smooth typing
                except Exception as e:
                    print(f"\n⚠️  Typing error: {e}")
                    break

    def stop_recording(self):
        """Stop recording and hand the buffer off to the transcription worker."""
        dbg("stop_recording called")
        if not self.is_recording:
            dbg("stop_recording: not recording, ignoring")
            return

        self.is_recording = False
        duration = time.time() - self.record_start_time

        # Hide the VU meter immediately on release — the visual should
        # disappear with the keypress, not after transcription finishes.
        if self._indicator is not None:
            self._send_to_indicator({"type": "hide"})

        # Restore system volume immediately so the user hears the beep at
        # their normal level.
        self.restore_system_volume()

        # Stop beep async — doesn't affect capture; mic stays open.
        threading.Thread(target=lambda: self.play_beep(self.STOP_BEEP_HZ, 0.1), daemon=True).start()

        print(f"⏹️  Stopped recording ({duration:.1f}s)")

        if self.real_time_mode:
            if self.transcription_thread:
                print("⏳ Finishing transcription...")
                sys.stdout.flush()
                self.transcription_thread.join(timeout=5.0)
                print("─" * 60 + "\n")
        else:
            if self.audio_data:
                self.transcribe_audio()
            else:
                print("❌ No audio recorded\n")

    def transcribe_audio(self):
        """Write the recorded audio to WAV and hand it off to the worker.

        Runs on the listener thread (because stop_recording is invoked from
        on_release). Doing the actual MLX call here would crash because MLX
        streams are per-thread — see _transcription_worker.
        """
        raw_bytes = b''.join(self.audio_data)
        sample_width = self.audio.get_sample_size(self.FORMAT)
        audio_seconds = len(raw_bytes) / (sample_width * self.CHANNELS * self.RATE)
        print(
            f"🎙️  Recorded {audio_seconds:.1f}s of audio "
            f"({len(raw_bytes) / 1024:.1f} kB, {len(self.audio_data)} chunks)",
            flush=True,
        )

        # Clean up the captured signal before the model sees it: kill
        # DC/rumble, notch out the start-beep bleed, and peak-normalize.
        # Debug audio (below) captures the post-processed version because
        # that's literally what the model heard — disable via config if
        # you need to inspect the raw mic signal.
        if self.audio_postprocess:
            t0 = time.time()
            try:
                raw_bytes = self._post_process_audio(raw_bytes)
                dbg(f"audio post-processing took {(time.time() - t0) * 1000:.1f}ms")
            except Exception as e:
                # Non-fatal: if filtering blows up we'd rather transcribe
                # the raw signal than drop the recording entirely.
                print(f"⚠️  Audio post-processing failed (using raw): "
                      f"{type(e).__name__}: {e}", flush=True)

        # Write to a temp WAV that the worker will transcribe and then delete.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
        try:
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(sample_width)
            wf.setframerate(self.RATE)
            wf.writeframes(raw_bytes)
            wf.close()
        except Exception as e:
            print(f"❌ Failed to write recording WAV: {type(e).__name__}: {e}", flush=True)
            return

        # Persist a copy under debug_audio/ so we can inspect what the model heard.
        if self.should_save_debug_audio:
            debug_path = self.debug_audio_dir / (
                f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            try:
                with open(temp_filename, 'rb') as src, open(debug_path, 'wb') as dst:
                    dst.write(src.read())
                print(f"🔍 Debug audio saved: {debug_path}", flush=True)
            except Exception as e:
                dbg(f"failed to write debug audio: {type(e).__name__}: {e}")

        dbg(f"enqueuing transcription job: {temp_filename}")
        self._transcription_queue.put(temp_filename)

    def _transcription_worker(self):
        """Dedicated thread that owns the MLX model and runs all inference.

        MLX GPU streams are per-thread, so the model must be loaded and used
        from one consistent thread. Producers (the recording flow) hand us
        WAV paths via self._transcription_queue.
        """
        dbg("transcription worker thread starting")
        try:
            self.load_model()
            self.warmup_model()
        except SystemExit:
            self._model_ready.set()
            return
        except Exception as e:
            print(f"❌ Worker failed during model setup: {type(e).__name__}: {e}", flush=True)
            if VERBOSE:
                import traceback
                traceback.print_exc()
            self._model_ready.set()
            return
        self._model_ready.set()
        dbg("transcription worker ready")

        while True:
            wav_path = self._transcription_queue.get()
            if wav_path is None:
                dbg("transcription worker got shutdown sentinel")
                break
            try:
                self._process_transcription(wav_path)
            except Exception as e:
                print(f"❌ Worker transcription error: {type(e).__name__}: {e}", flush=True)
                if VERBOSE:
                    import traceback
                    traceback.print_exc()
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def _process_transcription(self, wav_path):
        """Run inference on a WAV path and emit text + clipboard + paste."""
        print("⏳ Processing transcription...", flush=True)
        start_time = time.time()
        text = self._transcribe_file(wav_path)
        transcribe_time = time.time() - start_time

        if not text:
            print("⚠️  No speech detected in audio\n", flush=True)
            return

        print("\n" + "═" * 60)
        print("📝 TRANSCRIPTION")
        print("═" * 60)
        print(text)
        print("═" * 60)
        print(f"⏱️  Transcribed in {transcribe_time:.1f}s")

        try:
            pyperclip.copy(text)
            print("✅ Copied to clipboard")
        except Exception as e:
            print(f"⚠️  Could not copy to clipboard: {e}")

        if self.auto_paste:
            try:
                time.sleep(0.2)
                with self.kb_controller.pressed(Key.cmd):
                    self.kb_controller.press('v')
                    self.kb_controller.release('v')
                print("✅ Auto-pasted")
            except Exception as e:
                print(f"⚠️  Could not auto-paste: {e}")
                print("   Enable Accessibility permissions in System Preferences")

        print("─" * 60 + "\n", flush=True)

    def _spawn_indicator(self):
        """Launch indicator.py as a sidecar subprocess.

        We talk to it over its stdin as JSON lines (see indicator.py for the
        protocol). All failure modes here are non-fatal: if the spawn fails
        we just null out the handle and the rest of the app skips IPC. The
        recorder must work fully without the indicator.
        """
        if not self.show_indicator:
            dbg("indicator disabled via config (show_indicator: false)")
            return
        try:
            self._indicator = subprocess.Popen(
                [sys.executable,
                 str(Path(__file__).parent / 'indicator.py'),
                 '--style', self.indicator_style],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # line-buffered so each JSON line flushes immediately
            )
            dbg(f"indicator subprocess started (pid={self._indicator.pid}, style={self.indicator_style})")
        except Exception as e:
            print(f"⚠️  Indicator failed to start (non-fatal): "
                  f"{type(e).__name__}: {e}", flush=True)
            self._indicator = None

    def _send_to_indicator(self, msg):
        """Send one JSON message to the indicator. Silent on broken pipe.

        Called from many threads (listener for show/hide, capture loop for
        level, main for quit). subprocess.Popen.stdin.write isn't documented
        as thread-safe, but stdlib io.TextIOWrapper holds the GIL for the
        single write+flush sequence we do here — good enough for our rate.
        If the indicator died, BrokenPipeError nulls the handle so we stop
        trying.
        """
        proc = self._indicator
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            dbg(f"indicator pipe broken ({type(e).__name__}: {e}) — disabling")
            self._indicator = None

    def _get_indicator_anchor(self):
        """Return the (x, y) AppKit-coord top-left where the indicator
        should appear, or None if we can't compute one.

        Strategy:
          1. Ask the Accessibility API for the focused element's text
             caret rect. Land the indicator just below + slightly right
             of the caret. This is what the user wants in 95% of cases —
             the spectrum appears at the spot where their transcription
             is going to land.
          2. Fall back to NSEvent.mouseLocation() with a small offset
             from the I-beam.

        The AX path needs the same Accessibility permission already
        required for the simulated ⌘V auto-paste, so the permission cost
        is zero. Apps that don't publish their text caret correctly
        (Electron apps, some browsers' chrome) silently fall back to the
        mouse cursor.
        """
        caret = self._caret_top_left_appkit()
        if caret is not None:
            return caret
        return self._mouse_top_left_appkit()

    def _caret_top_left_appkit(self):
        """AX-based caret bounds → AppKit top-left, or None on any failure.

        Synchronous and runs on the listener thread. The AX call typically
        takes 1-20 ms; not free, but cheap enough to do on key press
        without the user noticing.
        """
        try:
            from ApplicationServices import (
                AXUIElementCreateSystemWide,
                AXUIElementCopyAttributeValue,
                AXUIElementCopyParameterizedAttributeValue,
                AXValueGetValue,
                kAXValueCGRectType,
                kAXFocusedUIElementAttribute,
                kAXSelectedTextRangeAttribute,
                kAXBoundsForRangeParameterizedAttribute,
            )
            from AppKit import NSScreen
        except Exception as e:
            dbg(f"AX import failed: {type(e).__name__}: {e}")
            return None

        try:
            sys_elem = AXUIElementCreateSystemWide()
            err, focused = AXUIElementCopyAttributeValue(
                sys_elem, kAXFocusedUIElementAttribute, None
            )
            if err != 0 or focused is None:
                dbg(f"AX focused-element lookup failed (err={err})")
                return None

            err, sel_range = AXUIElementCopyAttributeValue(
                focused, kAXSelectedTextRangeAttribute, None
            )
            if err != 0 or sel_range is None:
                dbg(f"AX selected-range lookup failed (err={err})")
                return None

            err, bounds_val = AXUIElementCopyParameterizedAttributeValue(
                focused, kAXBoundsForRangeParameterizedAttribute,
                sel_range, None,
            )
            if err != 0 or bounds_val is None:
                dbg(f"AX bounds-for-range lookup failed (err={err})")
                return None

            ok, rect = AXValueGetValue(bounds_val, kAXValueCGRectType, None)
            if not ok or rect is None:
                dbg("AX value unwrap failed")
                return None

            # rect is a CGRect in screen coords with origin = TOP-LEFT
            # (Quartz). Convert to AppKit (origin = BOTTOM-LEFT) using
            # the height of the screen containing the caret.
            caret_x = float(rect.origin.x)
            caret_top_y = float(rect.origin.y)
            caret_h = float(rect.size.height) or 18.0
            caret_bottom_y_quartz = caret_top_y + caret_h

            # Quartz→AppKit Y-flip uses the main screen's height — that's
            # the canonical macOS transform regardless of which physical
            # display the point lives on. Both coord spaces are anchored
            # to the main screen's corner; only the axis direction differs.
            main_h = self._main_screen_height()
            # AppKit y of where we want the indicator's top edge to sit:
            # just below the caret's bottom, with a small gap.
            top_y_appkit = main_h - caret_bottom_y_quartz - 4.0
            # Nudge a hair to the right so the panel sits next to (not
            # under) the caret column.
            top_x_appkit = caret_x + 4.0
            dbg(f"caret anchor: rect={rect}, top_left_appkit=({top_x_appkit:.0f}, {top_y_appkit:.0f})")
            return (top_x_appkit, top_y_appkit)
        except Exception as e:
            dbg(f"caret lookup failed: {type(e).__name__}: {e}")
            return None

    def _mouse_top_left_appkit(self):
        """NSEvent.mouseLocation() with a small below-right offset.

        Used only when the AX caret lookup fails (kAXErrorNoValue, an app
        that doesn't publish caret info, no focused text field, etc.).
        """
        try:
            from AppKit import NSEvent
            pt = NSEvent.mouseLocation()
            # +16 right, -8 below in AppKit coords (= below the I-beam tip).
            dbg(f"mouse fallback anchor: ({pt.x:.0f}, {pt.y:.0f})")
            return (float(pt.x) + 16.0, float(pt.y) - 8.0)
        except Exception as e:
            dbg(f"mouse fallback lookup failed: {type(e).__name__}: {e}")
            return None

    def _main_screen_height(self):
        """Height of the main screen in points, used for Quartz↔AppKit Y flip."""
        try:
            from AppKit import NSScreen
            main = NSScreen.mainScreen()
            if main is not None:
                return float(main.frame().size.height)
        except Exception as e:
            dbg(f"main-screen lookup failed: {type(e).__name__}: {e}")
        return 1080.0

    def _shutdown_indicator(self):
        """Tell the indicator to quit and reap it. Idempotent."""
        proc = self._indicator
        if proc is None:
            return
        self._send_to_indicator({"type": "quit"})
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            dbg("indicator didn't quit in time — killing")
            try:
                proc.kill()
            except Exception:
                pass
        self._indicator = None

    def _compute_spectrum_bands(self, samples_int16):
        """Log-spaced frequency band magnitudes (0..1) for one audio chunk.

        Output length = self._spectrum_band_count, which the active
        indicator style dictates (16 for 'eq', 32 for 'spectrogram').

        Pipeline:
          1. int16 → float32 in [-1, 1]
          2. Hann window (reduces spectral leakage so a single tone doesn't
             smear across multiple bands)
          3. rFFT → magnitude → power spectrum
          4. Mean power per log-spaced band edge (80 Hz .. 7.5 kHz)
          5. 10·log10 → dB
          6. Clamp [-30, +5] dB and linearly map to [0, 1]

        Band edges and the window are cached on first call (they depend
        only on chunk size, which is fixed). Subsequent calls are pure
        numpy math — ~70 µs on M-series for chunk_size=1024.
        """
        n = samples_int16.shape[0]
        if getattr(self, '_fft_n', None) != n:
            self._init_spectrum_cache(n)

        # Hann-windowed float spectrum, power = |X|^2.
        samples = samples_int16.astype(np.float32) * (1.0 / 32768.0)
        spec = np.fft.rfft(samples * self._fft_window)
        power = (spec.real * spec.real + spec.imag * spec.imag) + 1e-12

        # Mean power per band (vectorized via np.add.reduceat would be
        # marginally faster, but the BAND_COUNT-element Python loop here
        # is already negligible at 15 Hz).
        bands = np.empty(self._spectrum_band_count, dtype=np.float32)
        for i, (lo, hi) in enumerate(self._band_bins):
            bands[i] = power[lo:hi].mean() if hi > lo else power[lo]

        # Power → dB, clamped to a perceptually-useful display range.
        # Empirically calibrated on the built-in MBP mic at 16 kHz / 1024
        # chunk: deep silence registers ~-40 dB, room tone ~-30 dB, quiet
        # speech ~-5 dB, loud speech peaks ~+10 dB. Mapping [-30, +5] dB
        # → [0, 1] keeps silence dark, makes speech vibrant, and saturates
        # only on shouts.
        db = 10.0 * np.log10(bands)
        normalized = np.clip((db + 30.0) / 35.0, 0.0, 1.0)
        return normalized.tolist()

    # Empirical gain factor for the orb's waveform output. Built-in Mac
    # mics rarely send int16 peaks past ~8000 during normal speech, so
    # multiplying by 4/32768 maps "normal speech amplitude" to roughly
    # ±0.6 on the [-1, 1] scale the orb's deflection expects. Loud speech
    # saturates near ±1.0; whispers stay readable but subtle.
    _WAVEFORM_GAIN = 4.0

    def _compute_waveform_samples(self, samples_int16):
        """Downsample a chunk to self._waveform_length floats in [-1, 1].

        The orb draws each value as one point around a circle, so we don't
        need anti-aliased downsampling — simple decimation (every Nth
        sample) is visually identical at this resolution and avoids any
        FFT or filter pass.

        Output is plain Python list (JSON-serializable). At gain 4.0 with
        a typical built-in mic, normal speech produces ±0.4..0.7
        deflection on the orb; loud speech saturates near ±1.0.
        """
        n = samples_int16.shape[0]
        target = self._waveform_length
        if target <= 0 or n == 0:
            return [0.0] * max(target, 0)
        factor = max(1, n // target)
        decimated = samples_int16[::factor][:target]
        # Pad if for some reason we didn't get enough (shouldn't happen
        # with chunk_size=1024 and target=128).
        if decimated.shape[0] < target:
            decimated = np.pad(
                decimated, (0, target - decimated.shape[0])
            )
        normalized = np.clip(
            decimated.astype(np.float32) * (self._WAVEFORM_GAIN / 32768.0),
            -1.0, 1.0,
        )
        return normalized.tolist()

    def _init_spectrum_cache(self, n_samples):
        """Pre-compute Hann window + per-band FFT-bin index ranges.

        Called lazily on the first chunk (and only re-runs if chunk size
        ever changes, which it shouldn't).
        """
        freqs = np.fft.rfftfreq(n_samples, 1.0 / self.RATE)
        # 80 Hz - 7.5 kHz covers speech well; below 80 Hz is mostly
        # rumble / handling noise on a built-in mic, above 7.5 kHz is
        # near Nyquist for 16 kHz capture.
        f_min, f_max = 80.0, min(self.RATE * 0.47, 7500.0)
        edges = np.logspace(
            np.log10(f_min), np.log10(f_max),
            self._spectrum_band_count + 1,
        )
        bins = []
        for lo_f, hi_f in zip(edges[:-1], edges[1:]):
            i_lo = int(np.searchsorted(freqs, lo_f))
            i_hi = max(i_lo + 1, int(np.searchsorted(freqs, hi_f)))
            bins.append((i_lo, i_hi))
        self._band_bins = bins
        self._fft_window = np.hanning(n_samples).astype(np.float32)
        self._fft_n = n_samples
        dbg(f"spectrum cache initialized: n={n_samples}, bins={bins}")

    # Post-processing filter constants. Hand-tuned for 16 kHz speech:
    #   HP at 80 Hz   → kills AC hum / mic-handling thud, leaves all
    #                   speech intact (lowest voiced fundamentals
    #                   ≥ ~85 Hz on the deepest male voices).
    #   Notch Q=20   → -3dB bandwidth = 800/20 = 40 Hz. Wide enough
    #                   to catch the beep even if the speaker→air→mic
    #                   path detunes it slightly, narrow enough not to
    #                   carve a hole in nearby speech formants.
    #   Target -1 dBFS → headroom against post-filter clipping while
    #                    giving the model a consistent loudness regardless
    #                    of mic gain.
    _PP_HIGHPASS_HZ = 80.0
    _PP_NOTCH_Q = 20.0
    _PP_TARGET_DBFS = -1.0

    def _post_process_audio(self, raw_bytes):
        """Clean up int16 PCM audio before handing it to the model.

        Pipeline (zero-phase filtfilt → no time shift, no group delay):
          1. 4th-order Butterworth high-pass at _PP_HIGHPASS_HZ (rumble).
          2. Biquad notch at START_BEEP_HZ (suppresses the start-record
             beep bleed that the mic catches in the first ~100 ms).
          3. Peak-normalize to _PP_TARGET_DBFS.

        Cost: sub-millisecond on a few seconds of 16 kHz mono on M-series.

        scipy is imported lazily so the import-time cost only hits users
        who actually have post-processing enabled.
        """
        if not raw_bytes:
            return raw_bytes

        from scipy import signal as sps

        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # filtfilt needs at least (filter_order * 3 + 1) samples; for
        # very short recordings (< ~50 ms) just skip filtering and only
        # normalize. Better to ship the raw signal than crash.
        min_len_for_filtfilt = 64
        if samples.shape[0] >= min_len_for_filtfilt:
            sos_hp = sps.butter(
                4, self._PP_HIGHPASS_HZ,
                btype='highpass', fs=self.RATE, output='sos',
            )
            samples = sps.sosfiltfilt(sos_hp, samples)

            b_n, a_n = sps.iirnotch(
                float(self.START_BEEP_HZ),
                Q=self._PP_NOTCH_Q,
                fs=self.RATE,
            )
            samples = sps.filtfilt(b_n, a_n, samples)

        # Peak normalization. If the signal is essentially silent, leave
        # it alone (avoid blasting noise floor up to -1 dBFS).
        peak = float(np.max(np.abs(samples)))
        silence_floor = 0.005  # ≈ -46 dBFS — below this it's noise, not signal
        if peak > silence_floor:
            target = 10.0 ** (self._PP_TARGET_DBFS / 20.0)
            samples = samples * (target / peak)

        clipped = np.clip(samples, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()

    def _describe_key(self, key):
        """Return a human-readable description of a pynput key for debug logging."""
        try:
            if isinstance(key, KeyCode):
                return f"KeyCode(vk={key.vk}, char={key.char!r})"
            return f"Key.{getattr(key, 'name', repr(key))}"
        except Exception:
            return repr(key)

    def on_press(self, key):
        """Callback for key press events"""
        if VERBOSE:
            dbg(f"on_press: {self._describe_key(key)} (target={self._describe_key(self.record_key)})")
        if key == self.record_key:
            self.start_recording()

    def on_release(self, key):
        """Callback for key release events"""
        if VERBOSE:
            dbg(f"on_release: {self._describe_key(key)}")
        if key == self.record_key:
            self.stop_recording()

    def run(self):
        """Main run loop"""
        print("\n" + "═" * 60)
        if self.real_time_mode:
            print("🎤  VOICE TRANSCRIBER - REAL-TIME MODE")
        else:
            print("🎤  VOICE TRANSCRIBER")
        print("═" * 60)

        # Show configuration
        print(f"\n📋 Configuration:")
        print(f"   Model: {self.model_name}")
        if self.real_time_mode:
            print(f"   Mode: Real-time typing")
        else:
            print(f"   Auto-paste: {'Enabled' if self.auto_paste else 'Disabled'}")
        print(f"   Audio feedback: {'Enabled' if self.audio_feedback else 'Disabled'}")

        print("\n📖 Instructions:")
        print("   • Hold dictation key to record")
        if self.real_time_mode:
            print("   • Text will type in real-time as you speak")
        else:
            print("   • Release to transcribe and paste")
        print("   • Press Ctrl+C to quit")
        print("\n💡 Tip: Run detect_key.py to find your key code")

        # Check microphone access
        print("\n🔍 Checking microphone access...", flush=True)
        if self.check_microphone_access():
            print("✅ Microphone is accessible", flush=True)
        else:
            print("❌ Cannot access microphone!", flush=True)
            print("   Go to: System Preferences → Privacy & Security → Microphone", flush=True)
            print("   Enable access for Terminal (or your terminal app)", flush=True)
            sys.exit(1)

        # Open the mic ONCE and hold it open for the lifetime of the program.
        # Trades a persistent macOS recording-indicator for zero latency on
        # key press — the beep then truthfully signals "we're recording right
        # now" rather than firing during pyaudio's 100-300 ms cold open.
        try:
            dbg(f"opening persistent input stream (rate={self.RATE}, channels={self.CHANNELS}, chunk={self.CHUNK})")
            t_open = time.time()
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
            dbg(f"persistent input stream opened in {(time.time() - t_open) * 1000:.0f}ms")
        except Exception as e:
            print(f"❌ Failed to open mic stream: {type(e).__name__}: {e}", flush=True)
            print("   Check System Preferences → Privacy & Security → Microphone", flush=True)
            sys.exit(1)

        self._capture_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # MLX streams are per-thread, so spin up a dedicated worker that owns
        # the model from load through inference. Block here until it signals
        # ready so the user only sees the "Ready" banner once we can transcribe.
        print("", flush=True)
        self._worker_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        self._worker_thread.start()
        self._model_ready.wait()

        # Spawn the floating VU meter sidecar AFTER the model is ready so its
        # startup doesn't bottleneck the "Ready" banner, and BEFORE the
        # listener starts so the first keypress already has a live pipe.
        self._spawn_indicator()

        print("═" * 60, flush=True)
        print("✅ Ready! Press and hold the dictation key to start...", flush=True)
        print(f"   Hotkey: {self._describe_key(self.record_key)}", flush=True)
        print("═" * 60 + "\n", flush=True)

        # Start keyboard listener
        listener = None
        try:
            dbg("starting keyboard listener")
            listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
            )
            listener.start()

            # pynput's Quartz event tap on macOS swallows SIGINT before it
            # reaches the Python main thread, so install our own handler that
            # stops the listener and lets join() return.
            def _sigint(signum, frame):
                print("\n\n👋 Ctrl+C received, shutting down...\n", flush=True)
                self.is_recording = False
                self._capture_running = False
                self.restore_system_volume()
                self._shutdown_indicator()
                if listener is not None:
                    listener.stop()

            signal.signal(signal.SIGINT, _sigint)
            signal.signal(signal.SIGTERM, _sigint)

            dbg("keyboard listener running (join)")
            listener.join()
        except KeyboardInterrupt:
            # Belt-and-suspenders: if the signal does sneak through, still tidy up.
            print("\n\n👋 Exiting...\n", flush=True)
            self.is_recording = False
            self._capture_running = False
            self.restore_system_volume()
            self._shutdown_indicator()
            if listener is not None:
                listener.stop()
        except Exception as e:
            print(f"\n❌ Keyboard listener error: {type(e).__name__}: {e}", flush=True)
            print("   You may need to grant Accessibility permissions", flush=True)
            print("   Go to: System Preferences → Privacy & Security → Accessibility", flush=True)
            if VERBOSE:
                import traceback
                traceback.print_exc()

        # Cleanup
        self._capture_running = False
        self.restore_system_volume()
        self._shutdown_indicator()
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        except Exception as e:
            dbg(f"error closing stream on shutdown: {e}")
        if self.audio:
            self.audio.terminate()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Voice Transcriber - Push-to-talk terminal transcription tool'
    )
    parser.add_argument(
        '--real-time',
        action='store_true',
        help='Enable real-time typing mode (types as you speak)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug logging (also set via VERBOSE=1 env var)'
    )
    parser.add_argument(
        '--style', '--indicator-style',
        choices=['eq', 'spectrogram', 'orb'],
        default=None,
        help='Override indicator_style from config.yaml for this run only '
             '(eq, spectrogram, or orb)',
    )

    args = parser.parse_args()

    if args.verbose:
        globals()['VERBOSE'] = True
        print("🐛 Verbose logging enabled", flush=True)

    try:
        transcriber = VoiceTranscriber(
            config_path=args.config,
            real_time_mode=args.real_time,
            indicator_style_override=args.style,
        )
        transcriber.run()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
