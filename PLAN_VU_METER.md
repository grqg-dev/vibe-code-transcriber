# Plan: Floating VU Meter Indicator

**Branch:** `claude/vu-meter` (branched off `claude/parakeet-mlx`).
**Target:** Visual indicator that appears at the cursor when the user starts
recording, shows live mic level as a VU meter while they hold the hotkey, and
disappears on release. Lands at the same screen position where the transcribed
text will get auto-pasted.

Read [`CLAUDE.md`](CLAUDE.md) first for the codebase shape — threading model,
why MLX runs on its own worker, why the mic stays open. None of that should
change here.

---

## UX requirements

- **Pops up at the mouse cursor** on key press. Offset a bit (e.g. `+16, -32`)
  so it doesn't sit on top of the I-beam.
- **Updates ~15-30 fps** with live mic level — a horizontal strip of segmented
  bars, green → yellow → red, with peak-hold.
- **Disappears immediately on key release.**
- **Never steals focus.** This is non-negotiable: stealing focus breaks the
  ⌘V auto-paste that lands the transcription. macOS solves this exactly once,
  via `NSPanel` with `NSWindowStyleMaskNonactivatingPanel`.
- **Floats above all app windows.** `NSWindow.level = NSStatusWindowLevel`.
  Should still appear over fullscreen apps if reasonably possible
  (`collectionBehavior |= .fullScreenAuxiliary`).
- **Failure isolation.** If the indicator crashes or refuses to start, the
  transcriber must keep working — just without the visual.

---

## Architecture: sidecar process (recommended)

We considered an in-process PyObjC integration but rejected it: Cocoa requires
the main thread for `NSApp.run()`, and the recorder's main thread is currently
the cleanup-anchor for the listener / worker / capture threads. Refactoring
that would touch every control-flow path we just stabilized. The sidecar:

```
parent (transcribe.py)              child (indicator.py)
─────────────────────                ────────────────────
spawn at startup                      NSApp shared instance
                                      build NSPanel + custom NSView
                                      NSApp.run()  ← owns main thread
                                      reads JSON lines from stdin on a
                                        background thread, dispatches to
                                        main via dispatch_async()
on key press
  read cursor position
  write {"type":"show", "x":..., "y":...}\n  →   panel.setFrameOrigin(x,y)
                                                  panel.orderFrontRegardless()

_capture_loop  (every ~64 ms, while is_recording)
  compute RMS of the chunk
  write {"type":"level", "rms":0.42}\n        →   update bars (with decay)

on key release
  write {"type":"hide"}\n                     →   panel.orderOut(nil)

on exit
  write {"type":"quit"}\n                     →   NSApp.terminate(nil)
  process.wait(timeout=1)
```

**Why this is right:** parent stays single-threaded-clean; child gets a
dedicated main thread for Cocoa; if either side dies, the other carries on
or fails gracefully; the protocol is debuggable by running
`python indicator.py` and typing JSON by hand.

---

## IPC protocol — JSON lines over stdin

One message per line. UTF-8. Indicator reads with `sys.stdin.readline()` in
a background thread.

| Message | Fields | Direction | Notes |
|---|---|---|---|
| `show` | `x: float, y: float` | parent → child | AppKit coords (origin bottom-left). Position the panel top-left at these coords. |
| `level` | `rms: float` (0.0–1.0) | parent → child | Normalized level. Indicator decays the bar with `level = max(rms, level * 0.85)`. |
| `hide` | — | parent → child | Hide but keep the process alive. |
| `quit` | — | parent → child | Tear down NSApp and exit. |

The child does **not** talk back. Stdout is reserved for crash logs only.

If you need to extend (color theme, position offset, custom geometry) — add
fields, default-tolerate missing ones.

---

## Required changes in `transcribe.py`

Keep diff minimal. Don't refactor anything that isn't strictly required for
the indicator.

### 1. Spawn the indicator (in `__init__` or end of `run()`'s startup phase)

```python
self._indicator = None
if self.config.get('show_indicator', True):
    try:
        self._indicator = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / 'indicator.py')],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,   # silence indicator chatter
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,                   # line-buffered
        )
        dbg("indicator subprocess started")
    except Exception as e:
        print(f"⚠️  indicator failed to start: {e}", flush=True)
        self._indicator = None
```

Failure must be soft. Recorder keeps working without it.

### 2. `_send_to_indicator(msg: dict)` helper

```python
def _send_to_indicator(self, msg):
    if self._indicator is None or self._indicator.stdin is None:
        return
    try:
        self._indicator.stdin.write(json.dumps(msg) + "\n")
        self._indicator.stdin.flush()
    except (BrokenPipeError, OSError) as e:
        dbg(f"indicator pipe broken: {e}")
        self._indicator = None
```

If the pipe breaks (indicator died), null the handle and silently degrade.

### 3. In `start_recording` — show

Get cursor position **before** spawning the beep thread. PyObjC's
`AppKit.NSEvent.mouseLocation()` is callable from any thread on macOS and
returns AppKit coords directly. **Do not** shell out to `osascript` for this
— too slow.

```python
try:
    from AppKit import NSEvent
    pt = NSEvent.mouseLocation()
    self._send_to_indicator({"type": "show", "x": float(pt.x), "y": float(pt.y)})
except Exception as e:
    dbg(f"cursor location lookup failed: {e}")
```

### 4. In `_capture_loop` — send level (while is_recording)

Compute RMS from the int16 chunk and send it. Keep the math tight; this runs
60 fps at minimum and must not block the capture thread.

```python
if self.is_recording and self._indicator is not None:
    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size:
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        # int16 max = 32768; scale into 0..1 with some gain since voice
        # rarely peaks near full-scale on a built-in mic.
        level = min(rms / 32768.0 * 6.0, 1.0)
        self._send_to_indicator({"type": "level", "rms": level})
```

Throttle if perf turns out to be a problem (e.g., only send every 2nd chunk)
— but start without throttling and measure.

### 5. In `stop_recording` — hide

```python
self._send_to_indicator({"type": "hide"})
```

Right after the `is_recording = False` flip is fine.

### 6. Cleanup paths — quit and wait

In SIGINT handler, KeyboardInterrupt handler, and the end-of-`run()` cleanup
block:

```python
if self._indicator is not None:
    self._send_to_indicator({"type": "quit"})
    try:
        self._indicator.wait(timeout=1)
    except subprocess.TimeoutExpired:
        self._indicator.kill()
```

### 7. New config knob

Append to `config.yaml`:
```yaml
# Show a small VU meter at the cursor while recording.
show_indicator: true
```

And to `get_default_config()` in `transcribe.py`.

---

## New file: `indicator.py`

Single-file Python script, PyObjC-based. Lives at repo root next to
`transcribe.py`. Roughly 150-250 lines.

### Skeleton

```python
import sys, json, threading
from AppKit import (
    NSApp, NSApplication, NSPanel, NSView, NSColor, NSBezierPath, NSRect,
    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    NSStatusWindowLevel, NSBackingStoreBuffered,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
)
from PyObjCTools import AppHelper
from Foundation import NSObject, NSTimer
from Quartz import CGMainDisplayID  # only if you need display geometry

# --- VU view ---
class VUView(NSView):
    def initWithFrame_(self, frame):
        self = super().initWithFrame_(frame)
        self.level = 0.0
        self.peak = 0.0
        return self

    def setLevel_(self, value):
        # Called from main thread only (dispatched).
        self.level = max(value, self.level * 0.85)  # decay
        self.peak = max(self.peak * 0.97, self.level)
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        # 12 vertical bars, green/yellow/red gradient.
        # peak-hold line at self.peak.
        # rounded background.
        ...

# --- Panel ---
def make_panel():
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSRect((0, 0), (96, 28)),
        NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
        NSBackingStoreBuffered,
        False,
    )
    panel.setLevel_(NSStatusWindowLevel)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(NSColor.clearColor())
    panel.setHasShadow_(True)
    panel.setIgnoresMouseEvents_(True)
    panel.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorStationary
        | NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    view = VUView.alloc().initWithFrame_(NSRect((0, 0), (96, 28)))
    panel.setContentView_(view)
    return panel, view

# --- IPC reader thread ---
def reader(panel, view):
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except Exception:
            continue
        t = msg.get("type")
        if t == "show":
            AppHelper.callAfter(
                lambda: (panel.setFrameOrigin_((msg["x"] + 16, msg["y"] - 32)),
                         panel.orderFrontRegardless())
            )
        elif t == "level":
            AppHelper.callAfter(lambda: view.setLevel_(float(msg["rms"])))
        elif t == "hide":
            AppHelper.callAfter(lambda: panel.orderOut_(None))
        elif t == "quit":
            AppHelper.callAfter(lambda: NSApp().terminate_(None))
            return

if __name__ == "__main__":
    app = NSApplication.sharedApplication()
    panel, view = make_panel()
    threading.Thread(target=reader, args=(panel, view), daemon=True).start()
    AppHelper.runEventLoop()
```

**Don't use this skeleton verbatim** — verify each API against the installed
PyObjC version. `AppHelper.callAfter` is the idiomatic way to schedule a
callable on the main thread; `dispatch_async` would also work via
`Foundation.NSOperationQueue.mainQueue().addOperationWithBlock_`.

### Drawing details (`drawRect_`)

- Background: rounded rect, fill `NSColor.colorWithCalibratedWhite_alpha_(0.12, 0.9)`.
- 12 vertical bars, ~5 px wide, 2 px gap.
- Bars 0-7 green (`NSColor.systemGreenColor`), 8-9 yellow, 10-11 red.
- A bar at index `i` is lit if `i/12 <= level`.
- Peak-hold: 1-px white line at `peak * width`.

### Parent-death detection

The indicator must die when transcribe.py dies. Two options, pick one:

1. **Stdin EOF**: the `for line in sys.stdin` loop exits when the parent
   closes the pipe. Then call `NSApp.terminate_(None)`. Simplest.
2. **Poll PPID**: on macOS, the indicator's parent becomes `1` (launchd) if
   the original parent dies. Check `os.getppid() == 1` from a heartbeat
   thread.

Stdin EOF is enough.

---

## Dependencies

Add to `requirements.txt`:
```
pyobjc-framework-Cocoa
pyobjc-framework-Quartz
```

`pyobjc-framework-Cocoa` covers AppKit + Foundation, which is what we need.
`pyobjc-framework-Quartz` is needed if you reach for any Quartz/CG APIs
(e.g., screen geometry). If you don't end up using Quartz, drop it.

Total install size ≈ 30-40 MB. Pure Python wrappers over native frameworks
— no compilation step.

---

## Pitfalls — read these BEFORE you start

1. **Cocoa main-thread requirement.** All AppKit / NSPanel calls must be on
   the main thread of the indicator process. `AppHelper.callAfter()` is your
   bridge from the IPC reader thread.

2. **Coord systems.** `NSEvent.mouseLocation()` returns AppKit coords
   (origin bottom-left). `NSWindow.setFrameOrigin_()` also takes AppKit
   coords. As long as you stay in AppKit you don't need to flip. Don't mix
   with `CGEvent` coords (top-left).

3. **Don't use a normal `NSWindow`.** Use `NSPanel` with the
   `NSWindowStyleMaskNonactivatingPanel` bit, or focus *will* be stolen and
   auto-paste *will* break. Verify with the user's actual auto-paste flow,
   not just by eyeballing.

4. **`setIgnoresMouseEvents_(True)`.** Even non-activating panels can swallow
   clicks. Set this so the user can keep clicking through the meter.

5. **Don't block the capture thread.** The IPC write happens 15× per second
   from the capture thread. Use a non-blocking write or a `Queue` with a
   single drain thread between `_send_to_indicator` and the pipe if profiling
   shows blocking.

6. **Race at startup.** The indicator's NSPanel takes a few hundred ms to be
   fully ready. Early `show` messages may arrive before the reader thread is
   running. That's fine — the protocol is idempotent and lossy by design;
   the next `level` message will still cause the panel to appear correctly
   (you can call `orderFrontRegardless()` from any `level` message that
   arrives while the panel is hidden, or just live with a tiny first-press
   delay).

7. **Don't add another auto-paste recipient.** If the cursor moved between
   key press (when we record the show position) and the worker auto-pasting
   later, the paste lands at the *new* cursor, not where the indicator
   appeared. That's the existing macOS behavior and not your problem — but
   note it in the README so the user knows.

8. **Test with `attenuate_volume: false`** in config — beep volume changes
   while recording can be distracting when you're also debugging visuals.

---

## Suggested order of attack

1. **Get `indicator.py` rendering in isolation.** Hard-code a 0.5 level,
   launch it directly, verify panel appears, doesn't steal focus, is on top
   of other windows, ignores mouse clicks. **Do not** touch `transcribe.py`
   yet.

2. **Drive it from a shell.** `python indicator.py` and type
   `{"type":"show","x":500,"y":500}` followed by `{"type":"level","rms":0.8}`.
   Verify the bars animate.

3. **Wire spawn + show + hide into `transcribe.py`.** Skip levels at first —
   prove the show/hide cycle works on every key press without leaking
   processes.

4. **Hook up RMS in `_capture_loop`.** Watch with `-v` to confirm
   `_send_to_indicator` is being called at ~15 Hz.

5. **Polish.** Decay curve, peak hold, colors, position offset.

6. **Cleanup paths.** Verify quit-on-exit works for: normal Ctrl+C, killed
   parent (`kill -9` the recorder, confirm indicator dies within a second),
   parent crash during recording.

7. **Update README** — document the new config knob and the visual.

---

## Definition of done

- [ ] On hotkey press, an unobtrusive VU meter appears at the cursor within
      ~50 ms.
- [ ] Bars animate to match my voice level. Peak-hold visible.
- [ ] On release, meter disappears immediately.
- [ ] Auto-paste still lands transcription at the original cursor position.
- [ ] Indicator process never survives the recorder process.
- [ ] Recorder still works fully if indicator subprocess fails to start
      (kill `indicator.py`, confirm transcribe.py still records + transcribes
      cleanly).
- [ ] `show_indicator: false` in config disables it entirely.
- [ ] `./run.sh -v` shows the indicator-related debug lines.
- [ ] No new warnings, no leaked subprocesses (`ps aux | grep indicator`
      after `Ctrl+C` returns nothing).
- [ ] README + CLAUDE.md updated.

---

## Scope discipline — what NOT to do

- **No transcription-status indicator** in this pass. Recording-only.
- **No live-typing.** That's `--real-time` mode and a separate problem.
- **No clickable controls.** It's read-only.
- **No multi-monitor "follow cursor across displays"** logic — just place
  it at the cursor on key press and leave it there until release.
- **No replacing the beep.** The beep stays. The meter is additive.
- **No refactoring the threading model** of `transcribe.py`. If you find
  yourself touching the capture thread / worker thread / listener thread
  for more than just "compute RMS and call `_send_to_indicator`", stop and
  reconsider.
