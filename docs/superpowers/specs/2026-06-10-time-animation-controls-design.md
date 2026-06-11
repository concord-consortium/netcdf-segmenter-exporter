# Time Animation Controls — Design

**Date:** 2026-06-10
**Status:** Approved
**Scope:** Frontend only (`static/index.html`, `static/style.css`, `static/app.js`). No backend changes.

## Purpose

Let users animate through a dataset's time axis and fine-step it, instead of
dragging the slider. Requested controls: a play button plus forward/back nudge
buttons.

## UI

A compact control row in the Display section of the sidebar, directly under
the time slider:

```
[◀] [▶ (play/pause toggle)] [▶] [speed select]
```

- **Nudge back / nudge forward** (`#step-back-btn`, `#step-fwd-btn`): move the
  time index by ±1, clamped to `[0, count-1]`; update the label and re-render
  immediately.
- **Play/pause** (`#play-btn`): one button; its text/icon and `aria-label`
  toggle between play (▶) and pause (⏸).
- **Speed** (`#speed-select`): `Slow` = 1 step/sec, `Normal` = 4 steps/sec
  (default), `Fast` = 10 steps/sec. Values are stored as steps-per-second
  numbers in the option values. Changeable mid-playback (applies from the
  next tick).

## Playback behavior

- **Render-aware pacing.** Each tick: advance the slider one step, update the
  label, `await refreshOverlay()`, then wait the remainder of the speed
  interval (`max(0, 1000/stepsPerSec − renderTime)`). Playback never outruns
  the server; on slow files it degrades to as-fast-as-frames-render.
- **`refreshOverlay` returns a boolean** — false only on genuine failure
  (HTTP error, network error, no file/variable). A request superseded by a
  newer one returns **true**: a newer request owns the overlay, which is not
  a failure — otherwise a nudge during playback (which supersedes the play
  tick's render) would wrongly stop playback. This is the only change to
  existing code paths.
- **Stop conditions:** reaching the last step (button returns to ▶); pressing
  pause; a frame render returning false (prevents error-looping when the
  server is gone — the status line already shows the failure); the user
  dragging the slider (`input` event = user takes control); opening a new
  file (`applyMetadata` stops playback).
- **Restart:** pressing play while at the last step restarts from step 0.
- **Non-stopping interactions:** nudge buttons work during playback without
  stopping it; switching variables mid-play just changes what renders.
- **Cancellation:** an incrementing generation token (`playToken`); the async
  play loop exits when its token is stale. Pause/restart/new-file all bump
  the token. No `setInterval` (an interval could pile up fetches; the loop is
  strictly sequential).
- **No-op guards:** play and nudges do nothing when no file is open or the
  file has no time axis (`metadata.time` null or `count <= 1`).

## Why no backend work

The existing `GET /api/slice` ETag/304 revalidation makes replaying a range
nearly free after first render, and the per-request rendering (~130 ms for a
596×1385 grid) is within the Normal-speed budget.

## Error handling

All failure paths reuse the existing `status` line via `refreshOverlay`'s
internal try/catch; the player only consumes the boolean result.

## Testing

No JS unit-test framework exists in this project (deliberate). Verification
is the project's established Playwright browser pass: nudge forward/back
(label + overlay change, clamping at ends), play advances automatically and
the button toggles to pause, pause halts it, speed change takes effect,
slider drag stops playback, play at end restarts from 0, pytest suite (72)
untouched.
