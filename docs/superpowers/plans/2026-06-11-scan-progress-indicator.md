# Scan Progress Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a "Loading…" card with a real progress bar over the map while a variable's global color range is being scanned. Spec: `docs/superpowers/specs/2026-06-11-scan-progress-indicator-design.md`.

**Architecture:** The chunked range scan becomes a generator (`DatasetManager.iter_value_range`) yielding `(done, total)` per block; a new `GET /api/value-range` streams those as NDJSON lines (progress travels on the same response because handlers are serialized — a polled endpoint would be starved). The frontend pumps the stream into a `<progress>` bar overlaid on the map, then fetches the slice as usual. `/api/slice` behavior, ETags, and `value_range`'s contract are unchanged.

**Tech Stack:** Python/FastAPI StreamingResponse + pytest (TDD); vanilla JS ReadableStream reader; Playwright verification with the real nClimGrid file.

**Constraint for all tasks:** the user's own server may run on port 8000 — never start, stop, or use port 8000. Use port 8765 for live checks and kill it afterward. Current suite: 94 passed.

---

## File Structure

```
server/dataset.py       # _scan_range refactored onto new iter_value_range generator
server/app.py           # + GET /api/value-range (StreamingResponse, NDJSON)
static/index.html       # + #map-loading overlay inside #map
static/style.css        # + overlay styles (incl. [hidden] guard for flex)
static/app.js           # + showMapLoading/hideMapLoading/ensureValueRange; refreshOverlay calls it
tests/test_dataset.py   # + 3 generator tests
tests/test_api.py       # + 4 streaming-endpoint tests
```

---

### Task 1: `iter_value_range` generator (TDD)

**Files:**
- Modify: `server/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Add failing tests to `tests/test_dataset.py`** (the `_write_range_file` helper already exists there)

```python
def test_iter_value_range_reports_progress(tmp_path, monkeypatch):
    import server.dataset as dataset_module

    monkeypatch.setattr(dataset_module, "RANGE_SCAN_CHUNK", 1)
    m = DatasetManager()
    m.open(_write_range_file(tmp_path))  # 3 time steps -> 3 blocks
    progress = list(m.iter_value_range("v"))
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert m._ranges["v"] == (-5.0, 99.0)  # cached by the generator itself
    assert m.value_range("v") == (-5.0, 99.0)
    m.close()


def test_iter_value_range_empty_when_cached(sample_nc):
    m = DatasetManager()
    m.open(sample_nc)
    m.value_range("temperature")  # populate the cache
    assert list(m.iter_value_range("temperature")) == []
    m.close()


def test_iter_value_range_unknown_variable_raises(sample_nc):
    m = DatasetManager()
    m.open(sample_nc)
    with pytest.raises(KeyError):
        list(m.iter_value_range("nope"))
    m.close()
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/test_dataset.py -v`**

Expected: the 3 new tests FAIL with `AttributeError: 'DatasetManager' object has no attribute 'iter_value_range'`. Existing 30 pass. Observe before implementing.

- [ ] **Step 3: Refactor in `server/dataset.py`**

REPLACE the existing `_scan_range` method entirely with these two methods (the `value_range` method and `RANGE_SCAN_CHUNK` constant stay exactly as they are):

```python
    def iter_value_range(self, variable):
        """Yield (done_blocks, total_blocks) while computing and caching the
        variable's global value range. Yields nothing when already cached."""
        if self.ds is None:
            raise RuntimeError("No dataset is open")
        if variable not in self.ds.data_vars:
            raise KeyError(variable)
        if variable in self._ranges:
            return
        da = self.ds[variable]
        time = self.coords.get("time")
        if time and time in da.dims:
            steps = int(da.sizes[time])
            blocks = [
                da.isel({time: slice(i, i + RANGE_SCAN_CHUNK)})
                for i in range(0, steps, RANGE_SCAN_CHUNK)
            ]
        else:
            blocks = [da]
        vmin, vmax = np.inf, -np.inf
        total = len(blocks)
        for done, block in enumerate(blocks, start=1):
            values = np.asarray(block.values)  # native dtype: no float64 copy
            finite = values[np.isfinite(values)]
            if finite.size:
                vmin = min(vmin, float(finite.min()))
                vmax = max(vmax, float(finite.max()))
            yield done, total
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            self._ranges[variable] = (0.0, 1.0)
        else:
            self._ranges[variable] = (vmin, vmax)

    def _scan_range(self, variable):
        for _ in self.iter_value_range(variable):
            pass
        return self._ranges[variable]
```

(`_scan_range` keeps its signature — it is the cache-miss seam existing tests
monkeypatch — and now just consumes the generator. The generator writes the
cache itself so the streaming endpoint can use it directly; `value_range`'s
redundant re-assignment of the same tuple is harmless. Building the lazy
`blocks` list up front costs nothing — `isel` reads no data — and gives an
exact `total`.)

- [ ] **Step 4: Run the full suite**

```bash
.venv/bin/python -m pytest
```

Expected: `97 passed` — including ALL existing value_range tests (the cache seam and results are unchanged).

- [ ] **Step 5: Commit**

```bash
git add server/dataset.py tests/test_dataset.py
git commit -m "refactor: range scan as a progress-yielding generator"
```

End the commit message with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 2: `GET /api/value-range` streaming endpoint (TDD)

**Files:**
- Modify: `server/app.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Add failing tests to `tests/test_api.py`** (add `import json` to the imports at the top)

```python
def test_value_range_stream_progress_then_result(opened):
    res = opened.get("/api/value-range", params={"variable": "temperature"})
    assert res.status_code == 200
    lines = [json.loads(l) for l in res.text.strip().splitlines()]
    assert len(lines) >= 2  # at least one progress line, then the final line
    final = lines[-1]
    assert "vmin" in final and "vmax" in final
    dones = [s["done"] for s in lines[:-1]]
    for step in lines[:-1]:
        assert 1 <= step["done"] <= step["total"]
    assert dones == sorted(dones)
    # second request: range is cached -> only the final line
    res2 = opened.get("/api/value-range", params={"variable": "temperature"})
    lines2 = [json.loads(l) for l in res2.text.strip().splitlines()]
    assert len(lines2) == 1
    assert lines2[0] == final


def test_value_range_stream_unknown_variable_404(opened):
    res = opened.get("/api/value-range", params={"variable": "nope"})
    assert res.status_code == 404


def test_value_range_stream_requires_open_file(client):
    res = client.get("/api/value-range", params={"variable": "temperature"})
    assert res.status_code == 409


def test_value_range_stream_no_store_header(opened):
    res = opened.get("/api/value-range", params={"variable": "humidity"})
    assert res.headers["cache-control"] == "no-store"
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/test_api.py -v`**

Expected: the 4 new tests FAIL (404s from the static mount / wrong status codes). Existing 29 pass.

- [ ] **Step 3: Add the endpoint to `server/app.py`**

Add `import json` to the standard-library imports at the top, and extend the responses import to:

```python
from fastapi.responses import Response, StreamingResponse
```

Add the route after the `browse` endpoint (before the static mount):

```python
@app.get("/api/value-range")
async def value_range_stream(variable: str):
    """Stream range-scan progress as NDJSON, ending with the cached range.

    Progress can't come from a separately polled endpoint: handlers are
    serialized on the event loop, so a poll would be starved while the scan
    runs. It travels on this response instead, flushing one line per chunk.
    """
    _require_open()
    if variable not in manager.ds.data_vars:
        raise HTTPException(status_code=404, detail=f"Unknown variable: {variable}")

    async def stream():
        try:
            for done, total in manager.iter_value_range(variable):
                yield json.dumps({"done": done, "total": total}) + "\n"
            vmin, vmax = manager.value_range(variable)
            yield json.dumps({"vmin": vmin, "vmax": vmax}) + "\n"
        except Exception as exc:  # e.g. file replaced/removed between chunks
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )
```

- [ ] **Step 4: Run the full suite**

```bash
.venv/bin/python -m pytest
```

Expected: `101 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/app.py tests/test_api.py
git commit -m "feat: stream range-scan progress as NDJSON"
```

End with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 3: Map loading overlay (frontend)

**Files:**
- Modify: `static/index.html`
- Modify: `static/style.css`
- Modify: `static/app.js`

- [ ] **Step 1: Add the overlay markup in `static/index.html`**

Replace `<div id="map"></div>` with:

```html
  <div id="map">
    <div id="map-loading" hidden>
      <div id="map-loading-card">
        <span id="map-loading-text">Loading&#8230;</span>
        <progress id="map-loading-bar" max="1" value="0"></progress>
        <span id="map-loading-pct"></span>
      </div>
    </div>
  </div>
```

(Leaflet tolerates pre-existing children in its container; it appends its own panes alongside.)

- [ ] **Step 2: Append to `static/style.css`**

```css
#map { position: relative; }
#map-loading {
  position: absolute; inset: 0; z-index: 1500;
  display: flex; align-items: center; justify-content: center;
  pointer-events: none; /* the map stays interactive underneath */
}
#map-loading[hidden] { display: none; } /* flex would defeat the hidden attr */
#map-loading-card {
  background: rgba(255, 255, 255, 0.92); border: 1px solid #ddd;
  border-radius: 6px; padding: 14px 18px; display: flex; gap: 10px;
  align-items: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  font-size: 13px;
}
#map-loading-bar { width: 160px; }
#map-loading-pct { min-width: 3em; color: #555; }
```

- [ ] **Step 3: Add the stream pump to `static/app.js`**

Add after the `updateTimeLabel` function:

```javascript
function showMapLoading(done, total) {
  const bar = document.getElementById("map-loading-bar");
  bar.max = total;
  bar.value = done;
  setText("map-loading-pct", `${Math.round((done / total) * 100)}%`);
  document.getElementById("map-loading").hidden = false;
}

function hideMapLoading() {
  document.getElementById("map-loading").hidden = true;
}

async function ensureValueRange(variable, seq) {
  // Pump the NDJSON progress stream; show the bar only once a progress
  // line actually arrives (cached ranges emit just the final line).
  const res = await fetch(
    `/api/value-range?variable=${encodeURIComponent(variable)}`
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    setText("status", `Failed to compute value range: ${err.detail}`);
    return false;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      if (seq !== overlayRequestSeq) {
        reader.cancel();
        return false; // a newer request took over
      }
      const msg = JSON.parse(line);
      if (msg.error) {
        setText("status", `Failed to compute value range: ${msg.error}`);
        return false;
      }
      if (msg.total) showMapLoading(msg.done, msg.total);
    }
  }
  return true;
}
```

- [ ] **Step 4: Wire it into `refreshOverlay`**

In `refreshOverlay`, immediately after `const seq = ++overlayRequestSeq;` and the opening `try {`, insert:

```javascript
    if (!(await ensureValueRange(variable, seq))) {
      // stale (superseded) is not a failure; genuine errors are
      return seq !== overlayRequestSeq;
    }
```

and convert the function's existing `try { ... } catch { ... }` into
`try { ... } catch { ... } finally { ... }` with this finally block:

```javascript
  } finally {
    if (seq === overlayRequestSeq) hideMapLoading();
  }
```

(Hide only when this request is still the current one — a superseded
request must not hide the bar a newer scan is driving.)

- [ ] **Step 5: Verify**

```bash
node --check static/app.js          # expect: exit 0
.venv/bin/python -m pytest          # expect: 101 passed
```

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/style.css static/app.js
git commit -m "feat: progress bar over the map while scanning value ranges"
```

End with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 4: Browser verification (real file)

**Files:** none (verification only)

- [ ] **Step 1: Start a FRESH verification server on port 8765 (NOT 8000)** — fresh matters: the range cache is in-memory, and the bar only shows during an uncached scan.

```bash
lsof -ti:8765 | xargs kill 2>/dev/null; sleep 1
.venv/bin/python -c "
import uvicorn
from server.app import app
uvicorn.run(app, host='127.0.0.1', port=8765)
" > /tmp/ncse-progress.log 2>&1 &
sleep 3
```

- [ ] **Step 2: Browser checks (PASS/FAIL with evidence)**

Navigate to http://127.0.0.1:8765/, open
`data/nclimgrid_prcp.nc`
(1,577 steps → 25 scan chunks → ~14 s scan).

1. While the first slice loads: `#map-loading` is visible, the `<progress>`
   bar value advances over time (sample `bar.value/bar.max` twice ~3 s
   apart), and `#map-loading-pct` shows an increasing percentage.
2. When the scan finishes: the overlay hides and the prcp slice renders
   (overlay image present, legend shows the global range ≈ [0, 2620.7]).
3. Nudge forward: NO loading overlay (range cached), frame renders quickly.
4. Open the demo file `data/demo_global.nc`:
   first temperature render may flash the bar sub-second (2 chunks) — both
   "appears briefly" and "too fast to observe" are PASS; nudging afterward
   shows no overlay.
5. Console: zero JS errors.

- [ ] **Step 3: Teardown — MANDATORY**

```bash
lsof -ti:8765 | xargs kill 2>/dev/null; sleep 1; lsof -ti:8765 || echo clear
```

Close the browser; remove any artifacts created in the repo.

- [ ] **Step 4: No commit** (nothing changed); report results.
