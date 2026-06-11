# Fixed Color Range Per Variable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the slice color scale per variable to the true min/max across ALL time steps (exact, cached scan), so colors and the legend are comparable across the whole animation. Spec: `docs/superpowers/specs/2026-06-11-fixed-color-range-design.md`.

**Architecture:** `DatasetManager.value_range(variable)` returns the cached global range, computed once by `_scan_range` (chunked along time, memory-bounded, non-finite-excluding). `render_slice_png` gains optional explicit `vmin`/`vmax` (per-slice behavior unchanged when omitted). `/api/slice` wires them together after the ETag check (304s never pay the scan). `RENDER_VERSION` → 3 so version-salted ETags invalidate all per-slice-scaled cached PNGs. No frontend changes.

**Tech Stack:** Python/xarray/numpy (TDD with pytest); light Playwright check at the end.

**Constraint for all tasks:** the user's own server may run on port 8000 — never start, stop, or use port 8000. Use port 8765 for live checks and kill it afterward. Current suite: 85 passed.

---

## File Structure

```
server/dataset.py       # + RANGE_SCAN_CHUNK, DatasetManager.value_range/_scan_range/_ranges
server/rendering.py     # + vmin/vmax params; RENDER_VERSION 2 -> 3
server/app.py           # /api/slice passes the global range to the renderer
tests/test_dataset.py   # + 6 range tests
tests/test_rendering.py # + 2 explicit-range tests
tests/test_api.py       # + 1 fixed-range-across-time test
README.md               # limitation bullet replaced by fixed-scale note
```

---

### Task 1: `DatasetManager.value_range` (TDD)

**Files:**
- Modify: `server/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Add failing tests to `tests/test_dataset.py`**

Ensure `import pandas as pd` is among the imports at the top of the file (add it if missing), then append:

```python
def _write_range_file(tmp_path):
    """3 time steps; global min in step 0, global max in step 2, one inf."""
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    data = np.full((3, 2, 2), 10.0)
    data[0, 0, 0] = -5.0
    data[2, 1, 1] = 99.0
    data[1, 0, 1] = np.inf  # must be excluded from the range
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    path = tmp_path / "range.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def test_value_range_spans_all_time_steps_excluding_nonfinite(tmp_path):
    m = DatasetManager()
    m.open(_write_range_file(tmp_path))
    assert m.value_range("v") == (-5.0, 99.0)
    m.close()


def test_value_range_chunked_scan_matches(tmp_path, monkeypatch):
    import server.dataset as dataset_module

    # force one time step per block: min and max live in DIFFERENT blocks,
    # proving the cross-block reduction
    monkeypatch.setattr(dataset_module, "RANGE_SCAN_CHUNK", 1)
    m = DatasetManager()
    m.open(_write_range_file(tmp_path))
    assert m.value_range("v") == (-5.0, 99.0)
    m.close()


def test_value_range_caches_scan(sample_nc, monkeypatch):
    m = DatasetManager()
    m.open(sample_nc)
    calls = {"n": 0}
    real = m._scan_range

    def counting(variable):
        calls["n"] += 1
        return real(variable)

    monkeypatch.setattr(m, "_scan_range", counting)
    first = m.value_range("temperature")
    second = m.value_range("temperature")
    assert first == second
    assert calls["n"] == 1
    m.close()


def test_value_range_resets_on_new_open(sample_nc, rotated_nc):
    m = DatasetManager()
    m.open(sample_nc)
    m.value_range("temperature")
    m.open(rotated_nc)
    assert m._ranges == {}          # cache cleared with the old file
    lo, hi = m.value_range("temperature")
    assert lo < hi                  # recomputed for the new file
    m.close()


def test_value_range_all_nan_falls_back(tmp_path):
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.full((2, 2), np.nan))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    path = tmp_path / "allnan.nc"
    ds.to_netcdf(path)
    ds.close()
    m = DatasetManager()
    m.open(path)
    assert m.value_range("v") == (0.0, 1.0)
    m.close()


def test_value_range_unknown_variable_raises(sample_nc):
    m = DatasetManager()
    m.open(sample_nc)
    with pytest.raises(KeyError):
        m.value_range("nope")
    m.close()
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/test_dataset.py -v`**

Expected: the 6 new tests FAIL with `AttributeError: 'DatasetManager' object has no attribute 'value_range'` (and the chunked test additionally needs `RANGE_SCAN_CHUNK`). Existing 24 tests pass. Observe before implementing.

- [ ] **Step 3: Implement in `server/dataset.py`**

Add a module constant after the existing name-set constants:

```python
# time steps per block when scanning a variable's global value range:
# bounds memory (~210 MB per block for a 596x1385 float32 grid)
RANGE_SCAN_CHUNK = 64
```

In `DatasetManager.__init__`, add `self._ranges = {}` after the existing attributes. In `close()`, add `self._ranges = {}` after the line resetting `ds`/`coords`/`path` (open() calls close() before adopting a new file, so a successful open always starts with an empty cache, while a FAILED open leaves the previous file's cache intact).

Append these methods to `DatasetManager`:

```python
    def value_range(self, variable):
        """Global (vmin, vmax) of a variable across all time steps, cached.

        Non-finite values are excluded; an all-non-finite variable falls
        back to (0.0, 1.0). The first call per variable scans the whole
        variable (seconds for multi-GB files); later calls are cached.
        """
        if self.ds is None:
            raise RuntimeError("No dataset is open")
        if variable not in self.ds.data_vars:
            raise KeyError(variable)
        if variable not in self._ranges:
            self._ranges[variable] = self._scan_range(variable)
        return self._ranges[variable]

    def _scan_range(self, variable):
        da = self.ds[variable]
        time = self.coords.get("time")
        if time and time in da.dims:
            steps = int(da.sizes[time])
            blocks = (
                da.isel({time: slice(i, i + RANGE_SCAN_CHUNK)})
                for i in range(0, steps, RANGE_SCAN_CHUNK)
            )
        else:
            blocks = (da,)
        vmin, vmax = np.inf, -np.inf
        for block in blocks:
            values = np.asarray(block.values, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                vmin = min(vmin, float(finite.min()))
                vmax = max(vmax, float(finite.max()))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 0.0, 1.0
        return vmin, vmax
```

- [ ] **Step 4: Run the full suite**

```bash
.venv/bin/python -m pytest
```

Expected: `91 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/dataset.py tests/test_dataset.py
git commit -m "feat: cached global value range per variable on DatasetManager"
```

End the commit message with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 2: Explicit range in the renderer + RENDER_VERSION 3 (TDD)

**Files:**
- Modify: `server/rendering.py`
- Modify: `tests/test_rendering.py`

- [ ] **Step 1: Add failing tests to `tests/test_rendering.py`**

```python
def test_render_with_explicit_range_scales_and_clips():
    import io

    from matplotlib import image as mpimage

    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.array([[0.0, 5.0], [10.0, 20.0]]))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    coords = {"lat": "lat", "lon": "lon", "time": None}
    png, vmin, vmax = render_slice_png(ds, coords, "v", vmin=0.0, vmax=10.0)
    assert (vmin, vmax) == (0.0, 10.0)  # echoes the explicit scale
    rgba = mpimage.imread(io.BytesIO(png))
    top_left = rgba[0, 0, :3]       # lat=1 row: value 10.0 == vmax
    top_right = rgba[0, 1, :3]      # value 20.0 must CLIP to the same color
    bottom_right = rgba[-1, 1, :3]  # lat=0 row: value 5.0 -> mid-scale
    np.testing.assert_allclose(top_left, top_right, atol=0.01)
    assert top_left[0] > 0.9 and top_left[1] > 0.85   # viridis top = yellow
    assert abs(float(bottom_right[0]) - float(top_left[0])) > 0.2  # mid != top


def test_render_partial_explicit_range_falls_back_to_slice():
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    coords = {"lat": "lat", "lon": "lon", "time": None}
    _, vmin, vmax = render_slice_png(ds, coords, "v", vmin=0.0)  # vmax omitted
    assert (vmin, vmax) == (1.0, 4.0)  # half-specified ranges are ignored
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/test_rendering.py -v`**

Expected: both new tests FAIL — `TypeError: render_slice_png() got an unexpected keyword argument 'vmin'`. Existing 5 tests pass. Observe.

- [ ] **Step 3: Modify `server/rendering.py`**

Change the version constant (the comment above it already explains why):

```python
RENDER_VERSION = 3
```

Change the signature to:

```python
def render_slice_png(ds, coords, variable, time_index=0, cmap="viridis",
                     vmin=None, vmax=None):
```

Extend the docstring with: "When both vmin and vmax are given they define the
color scale (values outside clip to the end colors); when either is None the
scale is computed from this slice alone."

Replace the existing range-computation block (the `finite = ...` through the
`vmin, vmax = 0.0, 1.0` else) so the per-slice computation only runs when the
explicit range is incomplete — the surrounding lines (`data = np.array(...)`,
inf masking, and the `vmin == vmax` guard) stay exactly as they are:

```python
    if vmin is None or vmax is None:
        finite = data[~np.isnan(data)]
        if finite.size:
            vmin = float(finite.min())
            vmax = float(finite.max())
        else:
            vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1.0  # avoid a degenerate color scale
```

- [ ] **Step 4: Run the full suite**

```bash
.venv/bin/python -m pytest
```

Expected: `93 passed` — including the existing per-slice rendering tests, which must be untouched by the new parameters.

- [ ] **Step 5: Commit**

```bash
git add server/rendering.py tests/test_rendering.py
git commit -m "feat: explicit color range in the renderer; bump render version"
```

End with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 3: Wire `/api/slice` to the global range (TDD) + README

**Files:**
- Modify: `server/app.py`
- Modify: `tests/test_api.py`
- Modify: `README.md`

- [ ] **Step 1: Add a failing test to `tests/test_api.py`**

```python
def test_slice_value_range_fixed_across_time(opened):
    h0 = opened.get(
        "/api/slice", params={"variable": "temperature", "time_index": 0}
    ).headers
    h3 = opened.get(
        "/api/slice", params={"variable": "temperature", "time_index": 3}
    ).headers
    assert h0["x-vmin"] == h3["x-vmin"]
    assert h0["x-vmax"] == h3["x-vmax"]
    # global range over all 4 steps of uniform-[15,25) data
    lo, hi = float(h0["x-vmin"]), float(h0["x-vmax"])
    assert 15.0 <= lo < 16.0
    assert 24.0 < hi < 25.0
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/test_api.py -v`**

Expected: the new test FAILS — per-slice scaling makes `x-vmin`/`x-vmax`
differ between time_index 0 and 3 (random data; the per-slice extremes of two
different steps essentially never coincide exactly). Existing 28 tests pass.

- [ ] **Step 3: Modify the slice endpoint in `server/app.py`**

In `slice_png`, replace:

```python
    try:
        png, vmin, vmax = render_slice_png(
            manager.ds, manager.coords, variable, time_index
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
```

with:

```python
    try:
        # global per-variable scale: colors comparable across all time steps.
        # Computed after the ETag check so 304 responses never pay the scan.
        vmin, vmax = manager.value_range(variable)
        png, vmin, vmax = render_slice_png(
            manager.ds, manager.coords, variable, time_index,
            vmin=vmin, vmax=vmax,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
```

(The variable was already validated against `data_vars` above, so
`value_range` cannot raise KeyError here; `render_slice_png` echoes the
effective range, so the headers reflect any degenerate-scale guard.)

- [ ] **Step 4: Update `README.md`**

Remove this known-limitations bullet:

```markdown
- The slice color scale is computed per time step, so colors aren't
  comparable across time steps; the legend always reflects the current
  slice.
```

In "Using the app", change step 2 from:

```markdown
2. Pick a variable and scrub the time slider; the slice renders as a colored
   overlay (legend shows the slice's value range).
```

to:

```markdown
2. Pick a variable and scrub the time slider; the slice renders as a colored
   overlay. The color scale and legend are fixed per variable — the global
   min/max across all time steps — so colors are comparable as time changes.
   The first render of each variable computes that range (a brief delay on
   multi-GB files).
```

- [ ] **Step 5: Run the full suite**

```bash
.venv/bin/python -m pytest
```

Expected: `94 passed`.

- [ ] **Step 6: Commit**

```bash
git add server/app.py tests/test_api.py README.md
git commit -m "feat: slice colors use the variable's global range across time"
```

End with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 4: Light browser verification

**Files:** none (verification only)

- [ ] **Step 1: Start a verification server on port 8765 (NOT 8000)**

```bash
[ -f data/demo_global.nc ] || .venv/bin/python scripts/make_demo_data.py
lsof -ti:8765 | xargs kill 2>/dev/null; sleep 1
.venv/bin/python -c "
import uvicorn
from server.app import app
uvicorn.run(app, host='127.0.0.1', port=8765)
" > /tmp/ncse-range.log 2>&1 &
sleep 3
```

- [ ] **Step 2: Browser checks (PASS/FAIL with evidence)**

Navigate to http://127.0.0.1:8765/, open
`data/demo_global.nc`.

1. Note `#legend-min`/`#legend-max` text at time 0. Nudge forward 3 times:
   the legend text is IDENTICAL after every step (fixed range), while the
   overlay image URL changes (frames still render).
2. Click play, let it run ~3 seconds, pause: legend text still identical to
   the value from step 1.
3. Switch variable to `precipitation`: legend changes to that variable's
   global range; nudging again keeps it constant.
4. Console: zero JS errors.

- [ ] **Step 3: Teardown — MANDATORY**

```bash
lsof -ti:8765 | xargs kill 2>/dev/null; sleep 1; lsof -ti:8765 || echo clear
```

Close the browser. Never touch port 8000.

- [ ] **Step 4: No commit** (nothing changed); report results.
