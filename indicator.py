#!/usr/bin/env python3
"""
Sidecar visualization indicator for the Voice Transcriber.

Spawned as a subprocess by transcribe.py. Reads JSON-line messages from stdin
(one object per line) and drives a non-activating NSPanel that floats above
all windows. Three visualization styles, selected at spawn time:

    --style eq           16-band segmented VU/EQ bars with per-band peak hold.
                         Compact (200x40), responsive, classic look.
                         (Default — set in transcribe.py.)

    --style spectrogram  Scrolling waterfall, 32 freq bins × 80 time columns,
                         magma colormap. (240x72.)

    --style orb          Pulsing inner core surrounded by a circular
                         oscilloscope outline that wiggles with the live
                         waveform. (100x100, square.)

Why a sidecar process? Cocoa needs the main thread for NSApp.run(), but
transcribe.py's main thread is the cleanup anchor for the listener, capture,
and MLX worker threads. Boxing the whole AppKit lifecycle in its own process
keeps that contract intact and gives us a clean failure boundary: if the
indicator crashes, the recorder keeps working.

IPC — one JSON object per line on stdin:

    {"type":"show",     "x":FLOAT, "y":FLOAT}    # AppKit coords (origin = bottom-left).
                                                  # (x, y) = panel's TOP-LEFT. Parent
                                                  # computes placement (caret vs cursor).
    {"type":"spectrum", "bands":[FLOAT, ...]}    # Spectrum-domain payload, used by
                                                  # 'eq' (16 floats) and 'spectrogram'
                                                  # (32 floats). Values in [0, 1].
    {"type":"waveform", "samples":[FLOAT, ...]}  # Time-domain payload, used by 'orb'
                                                  # (128 floats). Values in [-1, 1].
    {"type":"hide"}                              # Hide panel + reset state, keep alive.
    {"type":"quit"}                              # Tear down NSApp and exit.

The parent (transcribe.py) MUST send the payload type that matches the
active style — views silently ignore payloads that aren't theirs.

Stdin EOF is also treated as quit — that's how we detect the parent dying.
Stdout is silent (parent redirects to DEVNULL anyway); stderr is reserved
for crash tracebacks.
"""

import argparse
import json
import math
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
from Foundation import NSMakePoint, NSMakeRect, NSObject
from PyObjCTools import AppHelper


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ Style #1: 16-band EQ (segmented VU bars)                              ║
# ╚════════════════════════════════════════════════════════════════════════╝
EQ_PANEL_W = 200
EQ_PANEL_H = 40
EQ_PADDING = 4
EQ_BAND_COUNT = 16
EQ_BAR_GAP = 2

# Stacked-bar color thresholds: bottom 65% of each bar is green, next 20%
# is yellow, top 15% is red. Classic hardware-VU look.
EQ_GREEN_FRAC = 0.65
EQ_YELLOW_FRAC = 0.20

# Smoothing curves. Levels arrive ~15 Hz from the capture loop; instant
# rise + decayed fall avoids strobing while still feeling responsive.
# Peak-hold falls slower so you can see your loudest moment per band.
EQ_LEVEL_DECAY = 0.70
EQ_PEAK_DECAY = 0.95


class SpectrumView(NSView):
    """Segmented 16-band EQ bars with per-band peak-hold."""

    def initWithFrame_(self, frame):
        self = objc.super(SpectrumView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._levels = [0.0] * EQ_BAND_COUNT
        self._peaks = [0.0] * EQ_BAND_COUNT
        return self

    def acceptPayload_(self, msg):
        bands = msg.get("bands")
        if not bands or len(bands) != EQ_BAND_COUNT:
            return
        for i, raw in enumerate(bands):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = 0.0
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            self._levels[i] = v if v > self._levels[i] else self._levels[i] * EQ_LEVEL_DECAY
            self._peaks[i] = max(self._peaks[i] * EQ_PEAK_DECAY, self._levels[i])
        self.setNeedsDisplay_(True)

    def reset(self):
        for i in range(EQ_BAND_COUNT):
            self._levels[i] = 0.0
            self._peaks[i] = 0.0
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bounds = self.bounds()

        NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.85).setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, 5.0, 5.0
        ).fill()

        inner_w = bounds.size.width - 2 * EQ_PADDING
        inner_h = bounds.size.height - 2 * EQ_PADDING
        total_gap = EQ_BAR_GAP * (EQ_BAND_COUNT - 1)
        bar_w = (inner_w - total_gap) / EQ_BAND_COUNT

        green_h = inner_h * EQ_GREEN_FRAC
        yellow_h = inner_h * EQ_YELLOW_FRAC

        green = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.30, 0.85, 0.35, 1.0)
        yellow = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.95, 0.80, 0.20, 1.0)
        red = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.95, 0.30, 0.25, 1.0)
        track = NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.06)
        peak_color = NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.85)

        for i in range(EQ_BAND_COUNT):
            x = EQ_PADDING + i * (bar_w + EQ_BAR_GAP)
            y0 = EQ_PADDING

            track.setFill()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, y0, bar_w, inner_h), 1.0, 1.0
            ).fill()

            lit = self._levels[i] * inner_h
            if lit > 0.5:
                green_lit = min(lit, green_h)
                green.setFill()
                NSBezierPath.bezierPathWithRect_(
                    NSMakeRect(x, y0, bar_w, green_lit)
                ).fill()

                if lit > green_h:
                    yellow_lit = min(lit - green_h, yellow_h)
                    yellow.setFill()
                    NSBezierPath.bezierPathWithRect_(
                        NSMakeRect(x, y0 + green_h, bar_w, yellow_lit)
                    ).fill()

                if lit > green_h + yellow_h:
                    red_lit = lit - green_h - yellow_h
                    red.setFill()
                    NSBezierPath.bezierPathWithRect_(
                        NSMakeRect(x, y0 + green_h + yellow_h, bar_w, red_lit)
                    ).fill()

            if self._peaks[i] > 0.04:
                peak_y = y0 + self._peaks[i] * inner_h
                peak_color.setStroke()
                line = NSBezierPath.bezierPath()
                line.moveToPoint_(NSMakePoint(x, peak_y))
                line.lineToPoint_(NSMakePoint(x + bar_w, peak_y))
                line.setLineWidth_(1.0)
                line.stroke()


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ Style #2: Scrolling spectrogram waterfall                             ║
# ╚════════════════════════════════════════════════════════════════════════╝
SG_PANEL_W = 240
SG_PANEL_H = 72
SG_PADDING = 4
SG_CORNER_RADIUS = 6.0
SG_ROWS = 32   # frequency bins
SG_COLS = 80   # time history columns (~5 sec at 15.6 Hz)

# Magma colormap control points (matplotlib's magma; perceptually uniform,
# dark-to-light). Pre-built into a 256-entry NSColor lookup at startup so
# drawRect_ never allocates a color object.
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
    """Interpolate _MAGMA_STOPS into a 256-entry NSColor lookup table."""
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
    """Scrolling waterfall: deque of frequency columns rendered as a 2-D
    grid of color-mapped cells."""

    def initWithFrame_(self, frame):
        self = objc.super(SpectrogramView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._cols = deque(maxlen=SG_COLS)
        for _ in range(SG_COLS):
            self._cols.append([0] * SG_ROWS)
        return self

    def acceptPayload_(self, msg):
        bands = msg.get("bands")
        if not bands or len(bands) != SG_ROWS:
            return
        col = [0] * SG_ROWS
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
        self._cols.clear()
        for _ in range(SG_COLS):
            self._cols.append([0] * SG_ROWS)
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bounds = self.bounds()

        NSColor.colorWithCalibratedWhite_alpha_(0.02, 0.92).setFill()
        bg_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, SG_CORNER_RADIUS, SG_CORNER_RADIUS
        )
        bg_path.fill()

        NSGraphicsContext.saveGraphicsState()
        bg_path.addClip()

        inner_w = bounds.size.width - 2 * SG_PADDING
        inner_h = bounds.size.height - 2 * SG_PADDING
        col_w = inner_w / SG_COLS
        row_h = inner_h / SG_ROWS
        cell_w = col_w + 0.6
        cell_h = row_h + 0.6

        lut = _COLOR_LUT
        bezier_rect = NSBezierPath.bezierPathWithRect_

        x = SG_PADDING
        for col in self._cols:
            y = SG_PADDING
            for idx in col:
                if idx > 1:
                    lut[idx].setFill()
                    bezier_rect(NSMakeRect(x, y, cell_w, cell_h)).fill()
                y += row_h
            x += col_w

        NSGraphicsContext.restoreGraphicsState()


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ Style #3: Orb (pulsing core + circular oscilloscope outline)          ║
# ╚════════════════════════════════════════════════════════════════════════╝
ORB_PANEL_W = 100
ORB_PANEL_H = 100
ORB_CORNER_RADIUS = 14.0       # softly rounded square background
ORB_WAVEFORM_LENGTH = 128      # MUST match parent's _waveform_length

# Geometry — chosen so the maximum-deflection waveform (radius RING + DEFL)
# stays well inside the rounded-rect's safe inscribed radius at 45°.
# Panel center = (PANEL_W/2, PANEL_H/2) = (50, 50). Safe-inscribed-radius
# at 45° given CORNER_RADIUS=14 ≈ panel_half − corner_r*(1 − 1/√2) ≈ 45.9 px.
# So MAX_RADIUS = ORB_RING_RADIUS + ORB_DEFLECTION = 44 ≤ 45.9. Safe.
ORB_INNER_RADIUS = 7           # base radius of the pulsing core
ORB_INNER_GROWTH = 10          # extra radius added at max amplitude
ORB_RING_RADIUS = 30           # baseline waveform circle
ORB_DEFLECTION = 14            # max wave deviation from baseline

ORB_WAVE_LINE_WIDTH = 1.4
ORB_AMP_DECAY = 0.80           # smoothing on the inner core's pulse size


class OrbView(NSView):
    """Pulsing inner core surrounded by a closed-loop circular oscilloscope.

    Each waveform sample is mapped to (angle, radius=baseline+sample*deflection)
    around the panel center. The result is a continuous closed curve that
    breathes with the audio. A separate inner core circle pulses in size
    with overall RMS (fast attack, slow decay).
    """

    def initWithFrame_(self, frame):
        self = objc.super(OrbView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._samples = [0.0] * ORB_WAVEFORM_LENGTH
        self._amp = 0.0
        # Pre-compute (cos, sin) per angle so the draw loop only does
        # arithmetic, no trig. Angle starts at 12 o'clock and goes
        # clockwise (negative y in math coords = upward in screen coords,
        # but since we're symmetric the direction is mostly cosmetic).
        self._unit_cos = [0.0] * ORB_WAVEFORM_LENGTH
        self._unit_sin = [0.0] * ORB_WAVEFORM_LENGTH
        for i in range(ORB_WAVEFORM_LENGTH):
            theta = 2.0 * math.pi * i / ORB_WAVEFORM_LENGTH - math.pi / 2.0
            self._unit_cos[i] = math.cos(theta)
            self._unit_sin[i] = math.sin(theta)
        return self

    def acceptPayload_(self, msg):
        samples = msg.get("samples")
        if not samples or len(samples) != ORB_WAVEFORM_LENGTH:
            return
        # Single pass: clamp samples in-place + accumulate sum-of-squares
        # for the RMS that drives the inner-core pulse.
        clamped = [0.0] * ORB_WAVEFORM_LENGTH
        rms_sq = 0.0
        for i, raw in enumerate(samples):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = 0.0
            if v < -1.0:
                v = -1.0
            elif v > 1.0:
                v = 1.0
            clamped[i] = v
            rms_sq += v * v
        self._samples = clamped
        rms = (rms_sq / ORB_WAVEFORM_LENGTH) ** 0.5
        # Fast attack, slow decay — matches the EQ smoothing feel.
        self._amp = rms if rms > self._amp else self._amp * ORB_AMP_DECAY
        self.setNeedsDisplay_(True)

    def reset(self):
        for i in range(ORB_WAVEFORM_LENGTH):
            self._samples[i] = 0.0
        self._amp = 0.0
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bounds = self.bounds()
        cx = bounds.size.width / 2.0
        cy = bounds.size.height / 2.0

        # Dark rounded background.
        NSColor.colorWithCalibratedWhite_alpha_(0.04, 0.90).setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, ORB_CORNER_RADIUS, ORB_CORNER_RADIUS
        ).fill()

        # Faint reference ring at the baseline waveform radius — gives
        # the eye an anchor to see how far the wave is deflecting.
        NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.05).setStroke()
        ref = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(
                cx - ORB_RING_RADIUS, cy - ORB_RING_RADIUS,
                ORB_RING_RADIUS * 2, ORB_RING_RADIUS * 2,
            )
        )
        ref.setLineWidth_(0.5)
        ref.stroke()

        # ── Inner glowing core ─────────────────────────────────────────
        # Three concentric circles of increasing opacity create a soft
        # bloom without needing NSShadow (which can be expensive on
        # continuous redraw). Color shifts warmer with amplitude.
        core_r = ORB_INNER_RADIUS + self._amp * ORB_INNER_GROWTH
        amp = self._amp
        # warm == cream-yellow; cool == soft white. Interpolate by amp.
        r_c = 1.00
        g_c = 0.95 - 0.10 * amp
        b_c = 0.85 - 0.30 * amp
        for r_mult, alpha in ((2.0, 0.06), (1.4, 0.18), (1.0, 0.95)):
            rr = core_r * r_mult
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                r_c, g_c, b_c, alpha
            ).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - rr, cy - rr, rr * 2, rr * 2)
            ).fill()

        # ── Circular oscilloscope ──────────────────────────────────────
        # Build the closed path: each sample contributes one point at
        # (angle_i, baseline + sample_i * deflection).
        path = NSBezierPath.bezierPath()
        cos_t = self._unit_cos
        sin_t = self._unit_sin
        baseline = ORB_RING_RADIUS
        defl = ORB_DEFLECTION
        for i, s in enumerate(self._samples):
            r = baseline + s * defl
            x = cx + r * cos_t[i]
            y = cy + r * sin_t[i]
            if i == 0:
                path.moveToPoint_(NSMakePoint(x, y))
            else:
                path.lineToPoint_(NSMakePoint(x, y))
        path.closePath()
        path.setLineWidth_(ORB_WAVE_LINE_WIDTH)

        # Two-pass stroke: a wider, semi-transparent stroke underneath
        # gives a soft glow without NSShadow.
        NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.35, 0.85, 0.95, 0.22
        ).setStroke()
        path.setLineWidth_(ORB_WAVE_LINE_WIDTH + 1.8)
        path.stroke()

        NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.55, 0.93, 1.00, 0.95
        ).setStroke()
        path.setLineWidth_(ORB_WAVE_LINE_WIDTH)
        path.stroke()


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ Style registry                                                         ║
# ╚════════════════════════════════════════════════════════════════════════╝
# Single source of truth for per-style panel geometry + view class. The
# parent (transcribe.py) mirrors this with its own per-style band/sample
# counts; both sides MUST stay in agreement.
STYLES = {
    "eq": {
        "panel_w": EQ_PANEL_W, "panel_h": EQ_PANEL_H,
        "view_class": SpectrumView,
    },
    "spectrogram": {
        "panel_w": SG_PANEL_W, "panel_h": SG_PANEL_H,
        "view_class": SpectrogramView,
    },
    "orb": {
        "panel_w": ORB_PANEL_W, "panel_h": ORB_PANEL_H,
        "view_class": OrbView,
    },
}


def _make_panel(style):
    """Build the non-activating floating panel for the given style.

    NSWindowStyleMaskNonactivatingPanel is the magic bit: it lets the
    panel appear on top without stealing focus, so the parent's ⌘V
    auto-paste still lands at the user's original cursor target.
    """
    cfg = STYLES[style]
    panel_w, panel_h = cfg["panel_w"], cfg["panel_h"]

    panel_style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, panel_w, panel_h),
        panel_style,
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

    view = cfg["view_class"].alloc().initWithFrame_(
        NSMakeRect(0, 0, panel_w, panel_h)
    )
    panel.setContentView_(view)
    return panel, view


class Controller(NSObject):
    """Owns the panel + view and exposes main-thread-safe operations.

    The stdin reader thread schedules these via AppHelper.callAfter so all
    AppKit mutation happens on the main thread (Cocoa is not thread-safe).

    Style-agnostic: each view implements acceptPayload_(msg) and pulls
    out the fields it cares about. The Controller is a dumb router.
    """

    def initWithPanel_view_height_(self, panel, view, panel_h):
        self = objc.super(Controller, self).init()
        if self is None:
            return None
        self.panel = panel
        self.view = view
        self.panel_h = panel_h
        return self

    def show_(self, point):
        """Show the panel anchored top-left at the given AppKit point.

        Parent computes placement (caret vs cursor); we just plant the
        panel there. NSWindow.setFrameOrigin_ takes the BOTTOM-left, so
        we subtract PANEL_H to land the top at the requested y.
        """
        x, y = point
        origin = NSMakePoint(float(x), float(y) - self.panel_h)
        self.panel.setFrameOrigin_(origin)
        self.view.reset()
        self.panel.orderFrontRegardless()

    def data_(self, msg):
        if not self.panel.isVisible():
            self.panel.orderFrontRegardless()
        self.view.acceptPayload_(msg)

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
            continue

        t = msg.get("type")
        if t == "show":
            AppHelper.callAfter(
                controller.show_, (msg.get("x", 0), msg.get("y", 0))
            )
        elif t == "spectrum" or t == "waveform":
            # The view itself decides whether the payload is its kind.
            AppHelper.callAfter(controller.data_, msg)
        elif t == "hide":
            AppHelper.callAfter(controller.hide_, None)
        elif t == "quit":
            AppHelper.callAfter(controller.quit_, None)
            return
        # Unknown types are ignored on purpose — forward-compatible protocol.

    # Stdin closed → parent gone.
    AppHelper.callAfter(controller.quit_, None)


def main():
    global _COLOR_LUT

    parser = argparse.ArgumentParser(description="Voice Transcriber indicator sidecar")
    parser.add_argument(
        "--style", choices=sorted(STYLES.keys()), default="eq",
        help="Visualization style (default: eq)",
    )
    args = parser.parse_args()

    app = NSApplication.sharedApplication()
    try:
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass

    # Build the magma LUT after NSApplication is initialized. Only the
    # spectrogram style actually uses it, but the cost is trivial (~256
    # NSColor allocations).
    _COLOR_LUT = _build_color_lut()

    panel, view = _make_panel(args.style)
    controller = Controller.alloc().initWithPanel_view_height_(
        panel, view, STYLES[args.style]["panel_h"]
    )

    threading.Thread(
        target=_reader_loop, args=(controller,), daemon=True
    ).start()

    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
