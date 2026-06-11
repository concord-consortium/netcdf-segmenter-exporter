# Fixed Color Range Per Variable — Design

**Date:** 2026-06-11
**Status:** Approved
**Scope:** `server/dataset.py` (range computation + cache), `server/rendering.py`
(optional explicit range + version bump), `server/app.py` (wire-up), tests,
README. No frontend changes.

## Purpose

The slice color scale is currently computed per time step, so the same value
gets different colors at different times and the legend flickers during
playback. Fix the scale per variable: vmin/vmax = the true minimum/maximum of
that variable across **all** time slices, computed exactly (no sampling, no
trusting possibly-stale CF `actual_range` attributes) and cached.

## `DatasetManager.value_range(variable)` (server/dataset.py)

- Returns `(vmin, vmax)` floats for a data variable of the open dataset.
- Computed by scanning the variable once in memory-bounded chunks along the
  time axis: `RANGE_SCAN_CHUNK = 64` time steps per block (~210 MB for a
  596×1385 float32 grid). Variables without a time dimension are reduced in
  one block.
- Non-finite values (NaN, ±inf) are excluded, consistent with rendering.
- All-non-finite variable → fall back to `(0.0, 1.0)`.
- (vmin == vmax is permitted here; rendering's existing `vmax = vmin + 1.0`
  degenerate-scale guard handles it.)
- Structure: `value_range(variable)` checks `self._ranges` and on a miss
  calls the private `self._scan_range(variable)` (which does the chunked
  reduction) and stores the result. This seam exists so tests can prove
  cache behavior by monkeypatching `_scan_range` with a counting wrapper.
- `self._ranges` is keyed by variable name; the cache is reset by `open()`
  and `close()` (it lives alongside `ds`/`coords`/`path`).
- Raises `KeyError` if the variable doesn't exist (callers validate first, as
  `/api/slice` already does) and `RuntimeError` if no dataset is open
  (matching `metadata()`).

## `render_slice_png(..., vmin=None, vmax=None)` (server/rendering.py)

- Two new optional keyword parameters. When BOTH are given, they define the
  color scale; slice values outside the range clip to the colormap's end
  colors (matplotlib's default normalize behavior). When either is None,
  the current per-slice computation runs unchanged (keeps the function
  independently testable and any direct callers unaffected).
- The existing inf→NaN masking and the `vmin == vmax` guard apply in both
  modes.
- The returned `(png, vmin, vmax)` echoes the effective scale used.
- `RENDER_VERSION` bumps to 3: rendered output changes for every slice, and
  the version-salted ETag (built for exactly this) invalidates all previously
  cached per-slice-scaled PNGs automatically.

## `/api/slice` (server/app.py)

- After the existing variable/time_index validation, call
  `manager.value_range(variable)` and pass the result to `render_slice_png`.
- `X-Vmin`/`X-Vmax` headers therefore carry the **global fixed range** — the
  legend becomes stable per variable with zero frontend changes.
- The first slice request for a variable pays the one-time scan (a few
  seconds for GB-scale files); subsequent requests hit the cache. No new
  status UI — the request simply takes longer once.
- ETag inputs are unchanged: the range is deterministic per (file, variable),
  which the key already covers; RENDER_VERSION=3 handles the upgrade.

## README

Replace the known-limitations bullet "The slice color scale is computed per
time step, so colors aren't comparable across time steps; the legend always
reflects the current slice." with a note under "Using the app": the color
scale and legend are fixed per variable (global min/max over all time
steps), and the first render of each variable computes that range — expect a
brief delay on multi-GB files.

## Testing

- **tests/test_dataset.py**: `value_range` returns the true global extremes
  across time steps (build a fixture where the global max lives in a later
  step than the global min); excludes inf; all-NaN variable → (0.0, 1.0);
  caching — wrap `manager._scan_range` with a counting monkeypatch and
  assert two `value_range` calls scan exactly once; cache cleared when a
  different file is opened (open file B → `value_range` scans again).
- **tests/test_rendering.py**: explicit `vmin`/`vmax` scale the colors — a
  cell AT vmax renders the colormap's top color even when the slice's own max
  is lower; out-of-range values clip rather than error; omitted params keep
  the old behavior (existing tests already lock this).
- **tests/test_api.py**: two requests for different `time_index` values
  return identical `X-Vmin`/`X-Vmax` equal to the variable's global range
  (the sample fixture's temperature spans ≈[15, 25) across 4 steps, so any
  single slice's range is strictly narrower — the assertion is
  discriminating).
- Browser pass (lightweight): play several frames; legend min/max text does
  not change between frames.
