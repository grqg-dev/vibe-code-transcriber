# 🎤 Voice Transcriber

A push-to-talk voice transcription tool for macOS on Apple Silicon. Hold a
key, speak, release — the transcription is dropped into your clipboard and
auto-pasted at the cursor. 100% local: speech never leaves the machine.

Powered by NVIDIA's Parakeet ASR models running natively on Apple Silicon via
[parakeet-mlx](https://github.com/senstella/parakeet-mlx) (MLX, not PyTorch —
fast cold start, no CUDA, no torch).

---

## ✨ Features

- **🎯 Push-to-talk** — Hold a hotkey to record, release to transcribe.
- **🔒 Fully local** — Model runs on your Mac's GPU/Neural Engine. No cloud.
- **📋 Clipboard + auto-paste** — Result is copied and pasted at the cursor.
- **🔊 Audio cue** — Start/stop beeps. Volume-attenuation knob for noisy meetings.
- **📊 Floating spectrogram waterfall** — A small live spectrogram (32
  frequency bins × ~5 sec of history, magma-colormapped) pops up next to
  the text insertion caret (or the mouse cursor, if no caret is exposed)
  while you hold the hotkey. See your voice's harmonics scroll past in
  real time and confirm at a glance that the mic is hot.
- **⚡ Zero key-press latency** — Mic is held open at startup; the beep is a
  truthful "now recording" cue, not a "starting to start" cue.
- **🔍 Debug audio dump** — Every recording is saved under `debug_audio/`
  so you can confirm exactly what the model heard.
- **🐛 Verbose mode** — `./run.sh -v` dumps detailed per-event diagnostics.

---

## 📦 Requirements

- **macOS on Apple Silicon** (M1 / M2 / M3 / …). The MLX runtime is Apple-only.
- **Python 3.9+** (tested on 3.13).
- **ffmpeg + portaudio** (installed by `install.sh` via Homebrew).
- **~1 GB free disk** for the model cache (`~/.cache/huggingface/`).
- **Microphone + Accessibility permissions** (macOS will prompt on first run).

---

## 🚀 Quick Start

```bash
cd vibe-code-transcriber
./install.sh    # one time: brews ffmpeg/portaudio, makes venv, pip installs
./run.sh        # daily driver
./run.sh -v     # same, with verbose diagnostics
```

First run downloads the model (~600 MB) and warms it up. Subsequent runs are
fast — the model is cached at `~/.cache/huggingface/`.

When you see `✅ Ready! Press and hold the dictation key to start...`, hold
the configured hotkey (default `alt_r`, i.e. right-option), speak, release.

---

## ⚙️ Configuration

All knobs live in [`config.yaml`](config.yaml). Defaults are sensible — only
the hotkey usually needs tweaking.

| Key | Default | What it does |
|---|---|---|
| `parakeet_model` | `mlx-community/parakeet-tdt-0.6b-v3` | HuggingFace model ID. See [models](#parakeet-models). |
| `hotkey_code` | `'alt_r'` | Either a pynput Key name (`'alt_r'`, `'f13'`, `'shift'`, `'ctrl'`, `'cmd'`) **or** a numeric VK code (e.g. `176` for the macOS Fn-dictation key, if you've remapped it). Use `python detect_key.py` to discover what your keys emit. |
| `auto_paste` | `true` | After transcription, simulate ⌘V at the current cursor. |
| `audio_feedback` | `true` | Play start/stop beeps. |
| `attenuate_volume` | `true` | Lower system **output** volume while recording (so background music/Zoom audio doesn't bleed into the mic). |
| `attenuation_percent` | `10` | Target % of original volume during recording. |
| `save_debug_audio` | `true` | Save every recording WAV under `./debug_audio/`. |
| `show_indicator` | `true` | Spawn the floating spectrogram sidecar (`indicator.py`) while recording. Disable if you find it distracting or if it ever misbehaves. |
| `audio.sample_rate` | `16000` | Parakeet expects 16 kHz mono. Don't change unless you know why. |
| `audio.channels` | `1` | Mono. |
| `audio.chunk_size` | `1024` | Frames per PortAudio chunk. 1024 / 16000 ≈ 64 ms per chunk. |

### Parakeet models

The `mlx-community` HuggingFace org hosts MLX-compatible Parakeet weights.

| Model | Size | Notes |
|---|---|---|
| `mlx-community/parakeet-tdt-0.6b-v3` | ~600 MB | **Default.** Best balance. |
| `mlx-community/parakeet-tdt-0.6b-v2` | ~600 MB | Older v2 weights. |
| `mlx-community/parakeet-rnnt-1.1b`   | ~1.1 GB | Larger, slower, marginally more accurate. |
| `mlx-community/parakeet-tdt_ctc-110m` | ~110 MB | Smallest, fastest, less accurate. |

All Parakeet models are **English-only**.

### Hotkey discovery

```bash
source venv/bin/activate
python detect_key.py     # press your key, copy what it prints into config.yaml
```

On macOS Sequoia/Tahoe, the right-option key reports as `Key.alt_r` (use
`hotkey_code: 'alt_r'`). The Fn-dictation key emits VK 176 only when explicitly
remapped under System Settings → Keyboard → Dictation.

---

## 🏗️ How it works (end-to-end)

```
┌───────────────────────── PROCESS ─────────────────────────┐
│                                                            │
│  main thread                                               │
│   └─ run()                                                 │
│       ├─ open persistent PyAudio input stream (16 kHz)     │
│       ├─ start _capture_thread  ──────────► [mic ring]     │
│       ├─ start _worker_thread   ──────────► [load model]   │
│       │      (blocks on _model_ready.wait())               │
│       ├─ start pynput keyboard.Listener (own thread)       │
│       └─ install SIGINT/SIGTERM → stop listener            │
│                                                            │
│  _capture_thread  (loops forever)                          │
│       └─ stream.read(CHUNK) every ~64 ms                   │
│            ├─ if is_recording: append to audio_data        │
│            └─ else: discard                                │
│                                                            │
│  pynput listener thread                                    │
│       ├─ on_press(key)                                     │
│       │     └─ if key == record_key: start_recording()     │
│       │           └─ flip is_recording = True              │
│       │              spawn async beep + volume attenuation │
│       └─ on_release(key)                                   │
│             └─ if key == record_key: stop_recording()      │
│                   ├─ flip is_recording = False             │
│                   ├─ restore system volume                 │
│                   ├─ spawn async stop-beep                 │
│                   └─ transcribe_audio()                    │
│                       ├─ join audio_data into WAV (temp)   │
│                       ├─ save copy to debug_audio/         │
│                       └─ enqueue path to _transcription_q  │
│                                                            │
│  _worker_thread (the only thread that touches MLX)         │
│       ├─ load_model() + warmup_model()                     │
│       ├─ _model_ready.set()                                │
│       └─ loop: pull WAV path from queue                    │
│            ├─ _transcribe_file() → text                    │
│            ├─ print, pyperclip.copy(text)                  │
│            ├─ simulate ⌘V if auto_paste                    │
│            └─ delete temp WAV                              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Why these design choices

1. **Persistent mic + capture thread.**
   pyaudio's `open(input=True)` blocks 100–300 ms on macOS. If the stream is
   opened *per key press*, the beep fires during this gap and you lose your
   first words. Holding the stream open from startup makes the keypress free
   — the beep becomes a truthful "we are recording RIGHT NOW" cue with a
   small inherent pre-roll buffer (~30 ms of audio captured before your
   keydown is included for free). Tradeoff: the macOS orange recording
   indicator stays lit while the app is running.

2. **Dedicated MLX worker thread.**
   MLX GPU streams are bound to the thread that created them. If you load
   the model on the main thread and then call `model.transcribe()` from
   pynput's listener thread, you crash with
   `RuntimeError: There is no Stream(gpu, 0) in current thread.`
   One worker thread owns load + inference; everyone else hands it a WAV
   path via a `queue.Queue`.

3. **Eager model load + warmup at startup.**
   The first inference is much slower than steady state. Loading and warming
   with 1 s of silence before showing the "Ready" banner means the user's
   first real recording feels just as fast as their hundredth.

4. **Shared PyAudio for beeps.**
   Creating a fresh `pyaudio.PyAudio()` for the beep would re-initialize
   CoreAudio and glitch the live input stream. The beep reuses
   `self.audio` and just opens a transient output stream on it.

5. **Hotkey accepts strings or VK codes.**
   On modern macOS, the right-option dictation key reports as `Key.alt_r`,
   not VK 176. Both formats are accepted so you can use whatever your
   hardware actually emits.

6. **Verbose mode + debug audio dump.**
   When something feels off ("did it record? did the key fire?"), `-v` gives
   you a play-by-play of every key event, stream open, beep, and worker
   action. The WAV in `debug_audio/` lets you literally listen to what the
   model heard.

7. **Spectrum indicator as a sidecar process.**
   Cocoa requires the main thread for `NSApp.run()`, but `transcribe.py`'s
   main thread is the cleanup anchor for the listener, capture, and worker
   threads. Putting the floating panel in its own subprocess
   (`indicator.py`) keeps that contract intact, talks to it over stdin as
   one-JSON-per-line messages, and gives us hard failure isolation: if the
   indicator crashes or the pipe breaks, recording continues. Closing the
   pipe (parent dying) also self-terminates the child, so we never leak
   indicator processes.

8. **Anchor at the text caret, not the mouse cursor.**
   On press the parent asks the macOS Accessibility API
   (`AXUIElementCopyParameterizedAttributeValue` with
   `kAXBoundsForRangeParameterizedAttribute`) for the focused element's
   selected-text bounds and places the indicator just below it. This is
   what you want 95% of the time: the spectrum appears at the spot where
   your transcription will land. Apps that don't publish caret info (some
   Electron apps, certain web text fields, an unfocused desktop) silently
   fall back to the mouse cursor. AX needs the same Accessibility
   permission already required for ⌘V auto-paste, so the permission cost
   is zero. Note: because the indicator is anchored once *at key press*,
   if you click somewhere else mid-recording the paste will land at the
   new cursor — that's existing macOS behavior and intentional.

9. **Spectrum is computed in the parent, rendered in the child.**
   The capture thread already has the raw int16 chunks; it runs an FFT
   (Hann-windowed, ~70 µs per 1024-sample chunk), bins into 32
   log-spaced bands from 80 Hz to 7.5 kHz, converts to dB, and ships the
   32 floats to the child as `{"type":"spectrum","bands":[...]}`. The
   child treats each message as a new rightmost column of the waterfall
   — no DSP, just deque maintenance + a 256-entry magma-colormap LUT
   lookup per cell + `NSBezierPath` rect fills. This split keeps PyObjC
   out of the audio path and numpy out of the AppKit process.

---

## ⌨️ Controls and CLI flags

| Action | How |
|---|---|
| Record | Hold the configured hotkey, release when done |
| Quit | `Ctrl+C` in the terminal (handled cleanly via signal handler) |
| Verbose logging | `./run.sh -v` or `VERBOSE=1 ./run.sh` |
| Real-time typing mode | `./run.sh --real-time` (experimental — see [caveats](#real-time-mode)) |
| Custom config | `./run.sh --config /path/to/other.yaml` |

### Real-time mode

`--real-time` types text into the active window *as you speak*, rather than
all at once on release. It's experimental: it transcribes accumulating audio
every ~1.5 s, then types only the diff. Quality is lower than waiting for the
full utterance, and it currently runs MLX from the transcription thread (which
may collide with the worker thread on long sessions). Use the default
push-to-talk mode for anything important.

---

## 🔒 Permissions

macOS will prompt for two permissions on first run. If you skip them, recording
or auto-paste will silently fail.

1. **Microphone** — System Settings → Privacy & Security → Microphone → enable
   for your terminal app (Terminal.app, iTerm, Ghostty, etc.).
2. **Accessibility** — System Settings → Privacy & Security → Accessibility →
   enable for your terminal app. Required for the simulated ⌘V auto-paste.

---

## 🤝 Sharing the mic with Zoom / other apps

CoreAudio supports multiple simultaneous readers — Zoom and Voice Transcriber
can both hold the mic open at once. Two caveats:

- **Volume attenuation** dampens **system output** while you record. In a Zoom
  call, that briefly quiets the other speakers. Set `attenuate_volume: false`
  in `config.yaml` before joining meetings.
- **Hotkey collisions** — if you've mapped right-option to "Press shortcut key
  twice for dictation" in macOS settings, your hotkey will also trigger that.
  Use a different key (e.g. `'f13'`) to avoid stomping.

---

## 🛠️ Build a standalone binary

```bash
./build.sh
```

PyInstaller bundles transcriber + parakeet-mlx + MLX runtime into
`dist/VoiceTranscriber`. Copy with `config.yaml` to another Apple Silicon Mac
and it runs without Python.

> ⚠️ The bundle is large and platform-locked to Apple Silicon. The model
> itself is downloaded on first run (still ~600 MB to `~/.cache/huggingface/`).

---

## 🐛 Troubleshooting

| Symptom | Try |
|---|---|
| No beep when pressing the key | The hotkey doesn't match. Run `python detect_key.py`, see what your key reports, set `hotkey_code` in `config.yaml`. Then `./run.sh -v` to confirm every press calls `on_press` with the expected key. |
| Cuts off the start of my speech | Should be fixed by the persistent-mic design — beep happens *after* recording is live. If still happening, verify with `-v`: `_capture_loop` should be appending chunks within ~30 ms of `start_recording`. |
| `RuntimeError: There is no Stream(gpu, 0) in current thread` | An MLX call slipped onto the wrong thread. All inference must go through `_transcription_worker`. Check that any new code routes WAVs through `_transcription_queue` rather than calling `_transcribe_file` directly. |
| `Ctrl+C` doesn't exit | The signal handler should stop the listener. If a future change breaks this, `pkill -9 -f transcribe.py` is the nuke option. |
| Auto-paste silently doesn't paste | Accessibility permission not granted. System Settings → Privacy & Security → Accessibility → enable for your terminal. |
| Mic indicator stays orange when I'm not recording | Yes, by design — the mic stream is held open for the program's lifetime to keep keypress latency at zero. Quit (`Ctrl+C`) to release it. |
| First-run model download stuck / slow | HuggingFace rate-limits unauthenticated users. Run `huggingface-cli login` (or set `HF_TOKEN`) for faster downloads. |
| Want to see what the model heard | Look in `debug_audio/` — each recording is saved as `recording_YYYYMMDD_HHMMSS.wav`. |
| Want to see exactly what's happening | `./run.sh -v` |

---

## 📁 Project layout

```
vibe-code-transcriber/
├── transcribe.py        # Main application. All logic lives here.
├── indicator.py         # Sidecar process: floating VU meter (NSPanel + PyObjC).
├── config.yaml          # Runtime configuration.
├── detect_key.py        # Standalone helper: print pynput Key/VK for any keypress.
├── install.sh           # Brew + venv + pip. One-time setup.
├── build.sh             # PyInstaller bundle → dist/VoiceTranscriber.
├── run.sh               # Activates venv and forwards args to transcribe.py.
├── requirements.txt     # Python deps (parakeet-mlx, pyaudio, pynput, …).
├── README.md            # This file.
├── CLAUDE.md            # Architecture / gotchas for agents and contributors.
└── debug_audio/         # Per-recording WAV dumps (gitignored, auto-created).
```

---

## 📄 License

MIT.

## 🙏 Credits

- [parakeet-mlx](https://github.com/senstella/parakeet-mlx) — Parakeet on Apple Silicon via MLX
- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for ML
- [NVIDIA Parakeet](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) — the ASR models
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) — mic input + speaker output
- [pynput](https://github.com/moses-palmer/pynput) — global hotkeys + synthetic keystrokes
- [PyInstaller](https://www.pyinstaller.org/) — standalone bundle
