#!/usr/bin/env python3
"""
Sidecar spectrogram indicator for the Voice Transcriber.

Spawned as a subprocess by transcribe.py. Reads JSON-line messages from stdin
(one object per line) and drives a non-activating NSPanel that floats above
all windows and shows a live scrolling spectrogram waterfall — frequency on
Y (low at bottom, high at top), time scrolling right→left, intensity mapped
through the magma colormap. Disappears the moment the parent says so.

Why a sidecar process? Cocoa needs the main thread for NSApp.run(), but
transcribe.py's main thread is the cleanup anchor for the listener, capture,
and MLX worker threads. Boxing the whole AppKit lifecycle in its own process
keeps that contract intact and gives us a clean failure boundary: if the
indicator crashes, the recorder keeps working.

IPC — one JSON object per line on stdin:

    {"type":"show",     "x":FLOAT, "y":FLOAT}    # AppKit coords (origin = bottom-left).
                                                  # (x, y) is where the panel's TOP-LEFT lands.
                                                  # Parent computes placement (caret vs cursor).
    {"type":"spectrum", "bands":[FLOAT, ...]}    # length must equal ROWS (32),
                                                  # each value normalized 0.0 .. 1.0.
                                                  # Pushed as a new rightmost column.
    {"type":"hide"}                              # hide panel + clear history, keep process alive.
    {"type":"quit"}                              # tear down NSApp and exit.

Stdin EOF is also treated as quit — that's how we detect the parent dying.
Stdout is silent (parent redirects to DEVNULL anyway); stderr is reserved
for crash tracebacks.
"""

import json
import sys
import threading
from collections import deque

import objc
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSGraphicsContext,
    NSPanel,
    NSStatusWindowLevel,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSMakeRect, NSObject
from PyObjCTools import AppHelper

# ── Geometry ─────────────────────────────────────────────────────────────
# Larger than the old VU panel because a spectrogram lives or dies on
# pixel density. Each cell is ~2 px wide × ~2 px tall — still tiny, but
# enough resolution to see harmonic stripes during speech.
PANEL_W = 240
PANEL_H = 72
PADDING = 4
CORNER_RADIUS = 6.0

ROWS = 32              # frequency bins; MUST equal _SPECTRUM_BAND_COUNT in transcribe.py
COLS = 80              # time history columns ≈ 5 seconds at 15.6 Hz (1024-sample chunks @ 16 kHz)

# ── Magma colormap ───────────────────────────────────────────────────────
# Control points lifted from matplotlib's magma colormap (perceptually
# uniform, dark-to-light, great for audio spectrograms). We pre-build a
# 256-entry NSColor lookup table at startup so drawRect_ never has to
# allocate a color object — that matters because we fill ROWS*COLS = 2560
# rects per frame, ~15 fps.
_MAGMA_STOPS = [
    (0.001, 0.000, 0.014),
    (0.039, 0.030, 0.117),
    (0.099, 0.067, 0.231),
    (0.171, 0.083, 0.328),
    (0.236, 0.094, 0.421),
    (0.319, 0.101, 0.479),
    (0.404, 0.108, 0.504),
    (0.490, 0.123, 0.510),
    (0.568, 0.138, 0.514),
    (0.654, 0.158, 0.497),
    (0.738, 0.182, 0.473),
    (0.821, 0.221, 0.435),
    (0.898, 0.272, 0.402),
    (0.957, 0.358, 0.382),
    (0.991, 0.480, 0.402),
    (0.997, 0.611, 0.457),
    (0.991, 0.738, 0.546),
    (0.987, 0.846, 0.653),
    (0.987, 0.991, 0.749),
]
_COLOR_LUT_SIZE = 256
_COLOR_LUT = None  # populated in main() once NSApp exists


def _build_color_lut():
    """Pre-allocate 256 NSColor objects evenly sampled along the magma stops.

    Allocating ~3000 NSColors per render frame would dominate CPU; doing it
    once at startup and indexing into an array is essentially free.
    """
    lut = []
    n_segments = len(_MAGMA_STOPS) - 1
    for i in range(_COLOR_LUT_SIZE):
        t = i / (_COLOR_LUT_SIZE - 1)
        pos = t * n_segments
        seg = min(int(pos), n_segments - 1)
        frac = pos - seg
        c0 = _MAGMA_STOPS[seg]
        c1 = _MAGMA_STOPS[seg + 1]
        r = c0[0] + (c1[0] - c0[0]) * frac
        g = c0[1] + (c1[1] - c0[1]) * frac
        b = c0[2] + (c1[2] - c0[2]) * frac
        lut.append(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 1.0)
        )
    return lut


class SpectrogramView(NSView):
    """Scrolling waterfall: rolling deque of frequency columns rendered as
    a 2-D grid of color-mapped cells.

    All state mutation happens on the main thread (dispatched via
    AppHelper.callAfter from the stdin reader). drawRect_ reads the same
    state, also from the main thread — no locking needed.
    """

    def initWithFrame_(self, frame):
        self = objc.super(SpectrogramView, self).initWithFrame_(frame)
        if self is None:
            return None
        # deque of ROWS-length lists. Oldest column at index 0 (drawn at
        # left), newest column at index COLS-1 (drawn at right). Pre-fill
        # with silent columns so the panel never shows half-empty.
        self._cols = deque(maxlen=COLS)
        for _ in range(COLS):
            self._cols.append([0] * ROWS)
        return self

    def appendColumn_(self, bands):
        """Push a new rightmost column of ROWS intensities (0..1 floats).

        We quantize to LUT indices here (0..255) instead of storing floats,
        because the drawing path is going to do that lookup anyway and
        this lets us skip a multiply + clamp + int conversion per cell
        per frame.
        """
        if not bands or len(bands) != ROWS:
            return
        col = [0] * ROWS
        for i, raw in enumerate(bands):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = 0.0
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            col[i] = int(v * (_COLOR_LUT_SIZE - 1))
        self._cols.append(col)
        self.setNeedsDisplay_(True)

    def reset(self):
        """Clear history — called on show/hide so a new recording starts blank."""
        self._cols.clear()
        for _ in range(COLS):
            self._cols.append([0] * ROWS)
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bounds = self.bounds()

        # Solid dark background — gives a sense of the panel's edges even
        # when the spectrogram is mostly empty.
        NSColor.colorWithCalibratedWhite_alpha_(0.02, 0.92).setFill()
        bg_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, CORNER_RADIUS, CORNER_RADIUS
        )
        bg_path.fill()

        # Clip subsequent drawing to the rounded rect so cells don't bleed
        # past the corners.
        NSGraphicsContext.saveGraphicsState()
        bg_path.addClip()

        inner_w = bounds.size.width - 2 * PADDING
        inner_h = bounds.size.height - 2 * PADDING
        col_w = inner_w / COLS
        row_h = inner_h / ROWS

        # The +0.6 over-fill on width/height eliminates sub-pixel gaps
        # between adjacent cells (col_w/row_h are typically ~2 px and not
        # an integer multiple of the backing-scale-factor pixel).
        cell_w = col_w + 0.6
        cell_h = row_h + 0.6

        lut = _COLOR_LUT
        # Bind locals for inner-loop speed. Each frame is 32*80 = 2560
        # iterations; PyObjC method calls are not cheap, so we hoist the
        # color-LUT and bezier-path constructor out of the loop.
        bezier_rect = NSBezierPath.bezierPathWithRect_

        x = PADDING
        for col in self._cols:
            # Row 0 = lowest frequency → drawn at the BOTTOM (low y in AppKit).
            y = PADDING
            for idx in col:
                # Skip near-silent cells: same color as the background, so
                # nothing to draw. Costs one int compare per cell, saves
                # the entire fill for the typical empty-panel case.
                if idx > 1:
                    lut[idx].setFill()
                    bezier_rect(NSMakeRect(x, y, cell_w, cell_h)).fill()
                y += row_h
            x += col_w

        NSGraphicsContext.restoreGraphicsState()


def _make_panel():
    """Build the non-activating floating panel that hosts the spectrogram.

    NSWindowStyleMaskNonactivatingPanel is the magic bit: it lets the panel
    appear on top without stealing focus, so the parent's ⌘V auto-paste
    still lands at the user's original cursor target.
    """
    style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, PANEL_W, PANEL_H),
        style,
        NSBackingStoreBuffered,
        False,
    )
    panel.setLevel_(NSStatusWindowLevel)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(NSColor.clearColor())
    panel.setHasShadow_(True)
    panel.setIgnoresMouseEvents_(True)
    panel.setHidesOnDeactivate_(False)
    panel.setBecomesKeyOnlyIfNeeded_(True)
    panel.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorStationary
        | NSWindowCollectionBehaviorFullScreenAuxiliary
    )

    view = SpectrogramView.alloc().initWithFrame_(NSMakeRect(0, 0, PANEL_W, PANEL_H))
    panel.setContentView_(view)
    return panel, view


class Controller(NSObject):
    """Owns the panel + view and exposes main-thread-safe operations.

    The stdin reader thread schedules these via AppHelper.callAfter so all
    AppKit mutation happens on the main thread (Cocoa is not thread-safe).
    """

    def initWithPanel_view_(self, panel, view):
        self = objc.super(Controller, self).init()
        if self is None:
            return None
        self.panel = panel
        self.view = view
        return self

    def show_(self, point):
        """Show the panel anchored top-left at the given AppKit point.

        The parent (transcribe.py) is responsible for computing exactly
        where the indicator should sit — caret-relative when AX gives us
        the text insertion point, mouse-relative when it doesn't. This
        method does no offset math: it places the panel's top-left at
        the requested (x, y).

        NSWindow.setFrameOrigin_ takes the BOTTOM-left of the frame, so
        we subtract PANEL_H to land the top at the requested y.
        """
        x, y = point
        from Foundation import NSMakePoint
        origin = NSMakePoint(float(x), float(y) - PANEL_H)
        self.panel.setFrameOrigin_(origin)
        self.view.reset()
        self.panel.orderFrontRegardless()

    def spectrum_(self, bands):
        # Race tolerance: if the very first 'spectrum' beats the 'show'
        # (or the user hid and is now updating again), make sure the
        # panel is up.
        if not self.panel.isVisible():
            self.panel.orderFrontRegardless()
        self.view.appendColumn_(bands)

    def hide_(self, _ignored=None):
        self.panel.orderOut_(None)
        self.view.reset()

    def quit_(self, _ignored=None):
        NSApp().terminate_(None)


def _reader_loop(controller):
    """Read JSON lines from stdin; dispatch each to the main thread.

    Stdin EOF — i.e. the parent closed the pipe or died — exits the loop
    and asks NSApp to terminate. That's our parent-death detector.
    """
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            # Malformed line: silently skip. Don't ever crash the UI for
            # a parent-side bug — degrade gracefully.
            continue

        t = msg.get("type")
        if t == "show":
            AppHelper.callAfter(
                controller.show_, (msg.get("x", 0), msg.get("y", 0))
            )
        elif t == "spectrum":
            AppHelper.callAfter(controller.spectrum_, msg.get("bands", []))
        elif t == "hide":
            AppHelper.callAfter(controller.hide_, None)
        elif t == "quit":
            AppHelper.callAfter(controller.quit_, None)
            return
        # Unknown types are ignored on purpose — forward-compatible protocol.

    # Stdin closed → parent gone. Tear down so we don't outlive them.
    AppHelper.callAfter(controller.quit_, None)


def main():
    global _COLOR_LUT
    app = NSApplication.sharedApplication()
    # Accessory == background agent UI element: no Dock icon, no menu bar,
    # no Cmd-Tab presence, and most importantly no brief focus-grab on the
    # very first orderFront call.
    try:
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass

    # Must build the LUT AFTER NSApplication.sharedApplication() — NSColor
    # depends on the app being initialized.
    _COLOR_LUT = _build_color_lut()

    panel, view = _make_panel()
    controller = Controller.alloc().initWithPanel_view_(panel, view)

    threading.Thread(
        target=_reader_loop, args=(controller,), daemon=True
    ).start()

    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
