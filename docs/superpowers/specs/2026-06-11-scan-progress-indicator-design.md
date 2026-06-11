# Scan Progress Indicator — Design

**Date:** 2026-06-11
**Status:** Approved
**Scope:** `server/dataset.py` (scan generator refactor), `server/app.py`
(streaming endpoint), `static/index.html`/`style.css`/`app.js` (map overlay
with progress bar), tests. No change to `/api/slice` behavior or ETags.

## Purpose

The first render of each variable scans the whole variable for its global
color range (~14 s on the 1.4 GB nClimGrid file) with no feedback. Show a
progress indicator over the map: "Loading…" with a real progress bar (the
scan is chunked, so true progress is known), per the user's preference for a
bar over a spinner.

## Constraint that shapes the design

Handlers are deliberately serialized on the event loop (netCDF4 is not
thread-safe), so a separately polled progress endpoint would be starved
while a scan blocks the loop. Instead the progress travels on the SAME
response as the scan: a streaming endpoint that yields a progress line
between chunks. Each yield returns control to the event loop, so bytes
flush incrementally and other requests can interleave between chunks —
never concurrently with one.

## Backend

### `DatasetManager.iter_value_range(variable)` (server/dataset.py)

- Generator. Validation identical to `value_range` (RuntimeError when
  nothing open; KeyError for unknown variable).
- If the range is already cached: yields nothing and returns immediately
  (callers see an empty stream → no progress UI flash).
- Otherwise scans block-by-block (same `RANGE_SCAN_CHUNK` logic, native
  dtype, finite-only), yielding `(done_blocks, total_blocks)` AFTER each
  block. On completion stores the result (with the all-non-finite →
  `(0.0, 1.0)` fallback) into `self._ranges`.
- `_scan_range(variable)` is reimplemented as: consume
  `iter_value_range(variable)` fully, then return `self._ranges[variable]`.
  Its signature and the `value_range` → `_scan_range` cache-miss seam are
  unchanged (existing tests that monkeypatch `_scan_range` keep passing).

### `GET /api/value-range?variable=<name>` (server/app.py)

- 409 when nothing open (`_require_open`); 404 for unknown variable
  (checked before streaming starts).
- Returns `StreamingResponse`, `media_type="application/x-ndjson"`. The
  generator yields one JSON line per progress step:
  `{"done": 7, "total": 25}\n`, and finally
  `{"vmin": <float>, "vmax": <float>}\n` (from `manager.value_range`, now
  cached). For an already-cached variable the stream is just the final line.
- Any exception mid-scan (e.g. the file was replaced/deleted between
  chunks) is caught inside the generator and emitted as a final
  `{"error": "<message>"}\n` line — a streaming response cannot change its
  status code after starting.
- Headers include `Cache-Control: no-store` (progress must never be cached).

## Frontend

### Markup/CSS

- `<div id="map-loading" hidden>` placed INSIDE the `#map` div (Leaflet
  tolerates extra children): a centered card with text span
  `#map-loading-text` ("Loading…"), a native
  `<progress id="map-loading-bar" max="1" value="0">`, and a percent label
  `#map-loading-pct`. Absolutely positioned, centered over the map,
  `z-index` above Leaflet panes (≥ 1000), semi-opaque white card with the
  map visible around it. `pointer-events: none` so the map stays
  interactive.

### app.js

- `ensureValueRange(variable, seq)` — fetches the stream, reads it
  incrementally (ReadableStream reader + TextDecoder, newline-split
  buffering), and:
  - on each `{done, total}` line: show `#map-loading` and set
    bar value/max and percent text (`Loading… 28%`). The overlay is shown
    only when a progress line arrives — cached variables never flash it.
  - on `{vmin, vmax}`: finish (the values themselves aren't needed —
    `/api/slice` headers remain the legend's source of truth).
  - on `{error}`: set the `status` line to the message and return false.
  - if `seq !== overlayRequestSeq` at any point: cancel the reader and
    return false (stale request).
  - network failure: status line message, return false.
- `refreshOverlay` calls `await ensureValueRange(variable, seq)` before the
  existing `/api/slice` fetch; a false return aborts the refresh (after the
  usual stale check). `#map-loading` is hidden in ALL exit paths of
  `refreshOverlay` (use try/finally).
- Playback needs no changes: the play loop already awaits `refreshOverlay`,
  so a scan simply makes that frame take longer, with the bar visible.

## Error handling

- Stream `{"error"}` lines and fetch failures surface on the existing
  `status` line; the overlay always hides.
- `/api/slice` itself is unchanged — even if `ensureValueRange` were
  skipped, the slice endpoint computes the range itself (the stream is an
  optimization for feedback, not a correctness dependency).

## Testing

- **tests/test_dataset.py**: `iter_value_range` yields strictly increasing
  `(done, total)` ending at `(total, total)` with the chunk forced to 1 via
  monkeypatched `RANGE_SCAN_CHUNK` (3-step file → exactly 3 yields); caches
  the same result `value_range` would produce; yields nothing when already
  cached; `_scan_range`/`value_range` behavior unchanged (existing tests).
- **tests/test_api.py**: streaming endpoint — parse the response text lines:
  ≥1 progress line then a final vmin/vmax line matching
  `manager.value_range`; second request for the same variable returns ONLY
  the final line (cached); 404 unknown variable; 409 when nothing open.
- **Browser pass** (real nClimGrid file on port 8765): on first prcp render
  the overlay appears with an advancing bar and percent, then disappears
  and the slice renders. The demo file (73 steps = 2 scan chunks) may flash
  the bar sub-second on each variable's FIRST render — acceptable; repeats
  (cached) must never show it.
