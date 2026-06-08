# CLAUDE.md

Push-to-talk voice transcriber, macOS Apple Silicon only. Single Python file
(`transcribe.py`) + YAML config. Default ASR: [FluidAudio][fluidaudio] in a
long-lived Swift sidecar (`asr_sidecar/`). Fallback: [parakeet-mlx][parakeet-mlx]
on the worker thread (MLX GPU). See [`README.md`](README.md) for the user view.

[fluidaudio]: https://github.com/FluidInference/FluidAudio
[parakeet-mlx]: https://github.com/senstella/parakeet-mlx

## Don't break these invariants

Each one was a real bug in a prior iteration.

1. **MLX runs only on the worker thread (`_transcription_worker`), and only
   when `asr_backend: mlx`.** Loading or calling the model from any other
   thread crashes with `RuntimeError: There is no Stream(gpu, 0) in current
   thread.` MLX binds GPU streams per-thread. To transcribe on MLX, **always**
   enqueue a WAV path on `self._transcription_queue` — never call
   `_transcribe_file` directly (except warmup on that same thread).

2. **One PyAudio instance, `self.audio`, for the whole process.** Don't
   construct a new `pyaudio.PyAudio()` anywhere (especially not in
   `play_beep`). A second PA session re-inits CoreAudio and visibly
   glitches the live input stream. The beep reuses `self.audio` with
   `output=True`.

3. **The mic input stream is opened once in `run()` and held open until
   exit.** Reopening per keypress costs 100-300 ms of dead air during which
   the beep already fired — early words get clipped. The tradeoff is a
   permanent macOS recording-indicator (intentional).

4. **`is_recording` is the sole gate on capture.** `_capture_loop` runs
   forever (until `_capture_running` flips false at shutdown), reading
   every ~64 ms; it appends to `audio_data` only when `is_recording` is
   True. Do not add a second flag.

5. **All spawned threads are `daemon=True`.** pynput's Quartz event tap on
   macOS can swallow SIGINT before it reaches the Python main thread.
   Daemon threads guarantee the process actually exits.

6. **Block on `self._model_ready.wait()` before showing "Ready".** Users
 must not be able to press the hotkey before ASR is ready: FluidAudio sidecar
 has emitted `{"type":"ready"}` on stdout, or MLX load + warmup finished.

7. **The visualization indicator is a sidecar process, never in-process.**
 `indicator.py` is spawned by `transcribe.py` (with `--style eq|spectrogram|orb`)
 and talks JSON-lines over stdin. Do not try to integrate AppKit into
 the parent: Cocoa requires the main thread for `NSApp.run()`, and the
 parent's main thread is the listener cleanup anchor. The protocol:

 ```
 {"type":"show",     "x":FLOAT,"y":FLOAT}     # AppKit top-left, computed by parent
 {"type":"spectrum", "bands":[F0..F_{n-1}]}   # used by eq (16) + spectrogram (32). [0,1].
 {"type":"waveform", "samples":[F0..F127]}    # used by orb. [-1, 1].
 {"type":"hide"}                              # hide panel + reset state
 {"type":"quit"}                              # tear down NSApp and exit
 ```

 Payload sizes are bound on both sides — `EQ_BAND_COUNT` / `SG_ROWS` /
 `ORB_WAVEFORM_LENGTH` in `indicator.py` MUST match `_spectrum_band_count`
 and `_waveform_length` derived from `indicator_style` in `transcribe.py`.
 New message types should be forward-compatible (unknown types are
 ignored). Indicator failures must always be soft — `_send_to_indicator`
 nulls the handle on `BrokenPipeError` and the rest of the app carries
 on. Adding a new style means: a new constants block + view class +
 STYLES entry in `indicator.py`, plus updating `__init__`'s style
 validator and the `_capture_loop` branch in `transcribe.py`.

8. **Indicator anchor: mode-selected, with mouse as default.** The parent
 computes the AppKit-coord top-left of the indicator in
 `_get_indicator_anchor` before sending `show`. `self.indicator_anchor`
 (config `indicator_anchor`, CLI `--anchor`, default `'mouse'`) picks
 the strategy:
 - `'mouse'` → straight to `NSEvent.mouseLocation()` (with a +16/-8
 offset so the panel doesn't sit on the I-beam). Works in every app,
 zero AX round-trip on the keypress hot path.
 - `'caret'` → try the macOS Accessibility API
 (`AXUIElementCopyParameterizedAttributeValue` →
 `kAXBoundsForRangeParameterizedAttribute`) for the focused element's
 selected-text bounds; place the panel just below it. Falls back to
 the mouse path if AX returns nothing.
 Quartz→AppKit Y-flip always uses the main screen's height. The
 indicator itself does NO offset math — it just sets `frameOrigin` to
 the requested point (minus PANEL_H for bottom-left framing). If you
 need to change the anchor logic, do it in the parent, not the child.
 Adding a new anchor mode means: a new branch in `_get_indicator_anchor`,
 plus updating `__init__`'s anchor validator and the `--anchor` CLI
 `choices=[...]`.

9. **FluidAudio inference only in `asr_sidecar` (Swift subprocess).** Python
 never calls CoreML. The sidecar is long-lived (models loaded once at startup,
 like `indicator.py`). IPC is JSON-lines:

 ```
 Parent → sidecar stdin:
 {"type":"transcribe", "id":1, "path":"/tmp/recording.wav"}
 {"type":"quit"}

 Sidecar → parent stdout:
 {"type":"ready"}
 {"type":"result", "id":1, "text":"...", "audio_s":49.5, "processing_s":0.6, "ok":true}
 {"type":"error", "id":1, "message":"...", "ok":false}
 ```

 A dedicated `_asr_stdout_reader` thread parses stdout and fulfills pending
 requests by `id`. The worker thread blocks on per-request `threading.Event`s
 (one inference at a time). Stdin EOF → sidecar exits. Sidecar spawn failures
 are fatal for `asr_backend: fluidaudio` (print `build_asr.sh` instructions).
 Real-time mode requires `asr_backend: mlx`.

## Commands

```bash
# parse + import sanity (always run after editing transcribe.py)
python3 -c "import ast; ast.parse(open('transcribe.py').read())"

# unit tests (fast, no models)
./run_tests.sh

# full suite including MLX + FluidAudio integration
./run_tests.sh --all

# run
./run.sh           # normal
./run.sh -v        # verbose: prints every keypress, beep, stream open, worker step

# escape hatch if a hung listener won't die
pkill -9 -f transcribe.py

# reproduce invariant #1 (MLX threading) — confirms it's still real
source venv/bin/activate && python3 <<'PY'
import threading
from parakeet_mlx import from_pretrained
m = from_pretrained('mlx-community/parakeet-tdt-0.6b-v3')
err = []
def x():
    try: m.transcribe('debug_audio/<any-existing>.wav')
    except Exception as e: err.append(type(e).__name__)
threading.Thread(target=x).start(); err and print(err)  # expect ['RuntimeError']
PY

# end-to-end MLX inference smoke (after install, not in CI)
source venv/bin/activate && python3 -c "
from parakeet_mlx import from_pretrained
r = from_pretrained('mlx-community/parakeet-tdt-0.6b-v3').transcribe('debug_audio/<any>.wav')
print(repr(r.text))
"

# build FluidAudio sidecar (requires Xcode 15+ / macOS SDK with C++ headers)
./build_asr.sh

# sidecar IPC smoke (after build + model download)
echo '{"type":"transcribe","id":1,"path":"debug_audio/<any>.wav"}' | ./asr_sidecar/.build/release/asr-sidecar --model-version v2
```

## File map

| File | Role |
|---|---|
| `transcribe.py` | The entire app: `VoiceTranscriber` class. |
| `asr_sidecar/` | Swift executable + FluidAudio SPM dep. Long-lived ASR; JSON IPC on stdin/stdout. Built by `build_asr.sh`. |
| `build_asr.sh` | `swift build -c release` for the ASR sidecar; prints binary path. |
| `indicator.py` | Sidecar process. NSPanel + one of three custom NSViews (eq / spectrogram / orb), selected by `--style` CLI arg. Reads `{"type":..., ...}` JSON lines from stdin. Self-terminates on stdin EOF (parent death). Does no DSP — only state maintenance + drawing. |
| `config.yaml` | Runtime config. Loaded once. |
| `detect_key.py` | Standalone: prints what pynput sees for any keypress. Use when the configured `hotkey_code` doesn't match what the user's key emits. |
| `install.sh` | brew + venv + pip + optional `build_asr.sh`. |
| `run.sh` | Auto-builds sidecar if missing; `venv/bin/python3 transcribe.py "$@"`. |
| `build.sh` | PyInstaller bundle into `dist/VoiceTranscriber`. Hidden-imports `parakeet_mlx`, `mlx`. Sidecar binary must ship alongside the app (not bundled here). |
| `debug_audio/` | Per-recording WAV dumps. Gitignored. |

Pytest suite in `tests/` — run `./run_tests.sh` (unit only) or
`./run_tests.sh --all` (includes MLX + FluidAudio integration). Committed dummy
audio lives in `tests/fixtures/audio/`; regenerate with
`python3 tests/fixtures/generate_audio_fixtures.py`.

## Threading model

```
main thread             listener thread        capture thread         worker thread
─────────────           ────────────────       ────────────────       ──────────────
run()                   on_press/on_release    _capture_loop          _transcription_worker
- open mic stream       - flip is_recording    - stream.read() loop   - load_model (MLX)
- spawn capture thread  - write WAV            - append if recording  - warmup_model
- spawn worker thread   - enqueue WAV path     - else discard         - spawn/load ASR (sidecar or MLX)
- start pynput listener                                                    - process queue forever
- install SIGINT
```

## Control flow

Press: `on_press` → `start_recording` flips `is_recording=True` and spawns a
fire-and-forget beep thread. The capture thread (already running) starts
appending chunks on its next ~64 ms read tick.

Release: `on_release` → `stop_recording` flips the flag off, restores
volume, fires stop-beep, calls `transcribe_audio` which writes a temp WAV,
copies it to `debug_audio/`, and puts the path on `_transcription_queue`.

Worker: pulls path → `_transcribe_via_sidecar` or `_transcribe_file` → prints, copies to clipboard,
simulates ⌘V (if `auto_paste`), deletes temp WAV. The `debug_audio/` copy
stays for later inspection.

Shutdown: SIGINT/SIGTERM handler flips `_capture_running=False` and calls
`listener.stop()`; `run()`'s cleanup closes the stream and terminates
PyAudio. Daemon threads die with the process.

## Known soft spots

- **Volume attenuation reset is fragile.** `restore_system_volume` only
  works if `original_volume` was captured at recording start, and there's
  no recovery if the process dies mid-recording. The user has flagged this
  as "fuckety." A fix probably caches the volume once at startup and always
  restores to that on stop / on exit / on SIGINT.
- **`_real_time_transcribe` violates invariant #1.** It calls
  `_transcribe_file` from its own thread instead of routing through the
  worker queue. It mostly works because the worker hasn't necessarily
  touched MLX yet when real-time loads it first, but it's fragile. Route
  it through the queue if you touch real-time mode.
- **PyInstaller bundle may miss MLX native dylibs.** `build.sh` uses
  `--collect-all mlx parakeet_mlx`; verify the binary actually loads
  before shipping.

## Conventions

- **One file.** Add functions to `transcribe.py`. Don't split.
- **Comments document WHY, not WHAT.** Existing code follows this. Match it.
- **No logging framework.** `print(..., flush=True)` for user output;
  `dbg(...)` for `--verbose` traces. Keep the emoji prefixes
  (`🎙️ ⏹️ 📝 🔍 🐛 ⚠️ ❌ ✅`) — they're part of the UX.
- **Heavy imports go inside `load_model`.** Keep argparse / `--help` snappy.
- **Don't add cloud fallback / API providers.** Being 100% local is the point.
- **Don't reformat the whole file.** Comment style and structure are deliberate.
- **Don't commit `debug_audio/`** (it's gitignored).
- **Don't `pip uninstall` leftover NeMo/torch from a prior branch.** Bloat
  but harmless. `rm -rf venv && ./install.sh` to clean.

## Pointers

- MLX: https://ml-explore.github.io/mlx/build/html/index.html
- parakeet-mlx: https://github.com/senstella/parakeet-mlx
- pynput macOS quirks: https://pynput.readthedocs.io/en/latest/limitations.html
- PyObjC (AppKit/NSPanel): https://pyobjc.readthedocs.io/en/latest/
- `NSWindowStyleMaskNonactivatingPanel` (why the indicator doesn't steal focus):
  https://developer.apple.com/documentation/appkit/nswindowstylemask/nonactivatingpanel
- AX caret bounds (`kAXBoundsForRangeParameterizedAttribute`):
  https://developer.apple.com/documentation/applicationservices/kaxboundsforrangeparameterizedattribute
