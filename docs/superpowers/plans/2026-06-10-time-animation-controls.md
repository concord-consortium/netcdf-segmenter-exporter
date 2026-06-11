# Time Animation Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add play/pause and ±1-step nudge controls (with a speed selector) to the time slider, per the approved spec at `docs/superpowers/specs/2026-06-10-time-animation-controls-design.md`.

**Architecture:** Frontend-only. The play loop is a sequential async loop (no `setInterval`) cancelled via an incrementing generation token; pacing is render-aware (advance, await the frame, then sleep the remainder of the speed interval). `refreshOverlay` gains a boolean return so the player can stop on genuine render failure without error-looping.

**Tech Stack:** Vanilla JS (static/app.js), HTML/CSS (static/index.html, static/style.css). No backend changes; pytest suite (72) must stay untouched. Verification via Playwright browser pass (the project's established approach for frontend behavior — there is deliberately no JS unit-test framework).

**Constraint for all tasks:** the user's own server runs on port 8000 — never start, stop, or use port 8000. Use port 8765 for any live checks and kill it afterward.

---

## File Structure

```
static/index.html   # + #time-controls row in the Display section (Task 1)
static/style.css    # + control-row styling (Task 1)
static/app.js       # refreshOverlay boolean + player logic + listeners (Task 2)
```

No new files. All element ids referenced by Task 2 are created in Task 1: `step-back-btn`, `play-btn`, `step-fwd-btn`, `speed-select`.

---

### Task 1: Control row markup and styling

**Files:**
- Modify: `static/index.html` (Display section)
- Modify: `static/style.css` (append)

- [ ] **Step 1: Add the control row to `static/index.html`**

In the Display `<section>`, insert this block BETWEEN the time-step `<label>` (the one wrapping `#time-slider`) and the `<div id="legend">`:

```html
      <div id="time-controls">
        <button id="step-back-btn" type="button" aria-label="Back one time step">&#9664;</button>
        <button id="play-btn" type="button" aria-label="Play">&#9654; Play</button>
        <button id="step-fwd-btn" type="button" aria-label="Forward one time step">&#9654;</button>
        <select id="speed-select" aria-label="Playback speed">
          <option value="1">Slow</option>
          <option value="4" selected>Normal</option>
          <option value="10">Fast</option>
        </select>
      </div>
```

(Option values are steps-per-second numbers, read directly by the player.)

- [ ] **Step 2: Append to `static/style.css`**

```css
#time-controls {
  display: flex; gap: 6px; align-items: center; margin-top: 4px;
}
#time-controls button { margin-top: 0; padding: 4px 10px; }
#time-controls #play-btn { flex: 1; white-space: nowrap; }
#time-controls select { width: auto; flex: 0 0 auto; padding: 4px; }
```

(The `#time-controls select` rule overrides the global `select { width: 100% }` by specificity.)

- [ ] **Step 3: Verify serving and suite**

```bash
.venv/bin/python -m pytest          # expect: 72 passed
grep -c 'id="step-back-btn"\|id="play-btn"\|id="step-fwd-btn"\|id="speed-select"' static/index.html   # expect: 4
```

- [ ] **Step 4: Commit**

```bash
git add static/index.html static/style.css
git commit -m "feat: markup and styling for time animation controls"
```

End the commit message with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 2: Player logic in app.js

**Files:**
- Modify: `static/app.js`

- [ ] **Step 1: Make `refreshOverlay` return a boolean**

Apply exactly these changes inside the existing `refreshOverlay`:

- `if (!metadata) return;` → `if (!metadata) return false;`
- `if (!variable) return;` → `if (!variable) return false;`
- first stale check `if (seq !== overlayRequestSeq) return;` →
  `if (seq !== overlayRequestSeq) return true; // a newer request owns the overlay — not a failure`
- `if (!res.ok) { setText("status", "Failed to render slice."); return; }` → same but `return false;`
- second stale check (after `await res.blob()`) → `return true;` with the same comment
- at the very end of the `try` block (after the overlay create/update if/else), add `return true;`
- in the `catch` block, after `setText(...)`, add `return false;`

- [ ] **Step 2: Add player state and functions**

After the `updateTimeLabel` function, add:

```javascript
let playing = false;
let playToken = 0;

function timeStepCount() {
  return metadata && metadata.time ? metadata.time.count : 0;
}

function setTimeIndex(idx) {
  document.getElementById("time-slider").value = String(idx);
  updateTimeLabel();
}

async function nudge(delta) {
  const count = timeStepCount();
  if (count <= 1) return;
  const slider = document.getElementById("time-slider");
  const idx = Math.min(count - 1, Math.max(0, Number(slider.value) + delta));
  if (idx === Number(slider.value)) return;
  setTimeIndex(idx);
  await refreshOverlay();
}

function stopPlayback() {
  playing = false;
  playToken += 1; // cancels any in-flight play loop
  const btn = document.getElementById("play-btn");
  btn.textContent = "▶ Play";
  btn.setAttribute("aria-label", "Play");
}

async function startPlayback() {
  const count = timeStepCount();
  if (count <= 1) return;
  const token = ++playToken;
  playing = true;
  const btn = document.getElementById("play-btn");
  btn.textContent = "⏸ Pause";
  btn.setAttribute("aria-label", "Pause");

  const slider = document.getElementById("time-slider");
  if (Number(slider.value) >= count - 1) {
    setTimeIndex(0); // play pressed at the end: restart from the start
    const ok = await refreshOverlay();
    if (token !== playToken) return;
    if (!ok) {
      stopPlayback();
      return;
    }
  }

  while (token === playToken) {
    const started = performance.now();
    const idx = Number(slider.value);
    if (idx >= count - 1) break; // reached the end
    setTimeIndex(idx + 1);
    const ok = await refreshOverlay();
    if (token !== playToken) return; // paused or restarted while rendering
    if (!ok) break; // genuine render failure: stop, don't error-loop
    const stepsPerSec = Number(document.getElementById("speed-select").value);
    const dwell = Math.max(0, 1000 / stepsPerSec - (performance.now() - started));
    await new Promise((resolve) => setTimeout(resolve, dwell));
  }
  if (token === playToken) stopPlayback();
}

function togglePlayback() {
  if (playing) stopPlayback();
  else startPlayback();
}
```

- [ ] **Step 3: Stop playback on new file and on manual slider drag; wire the buttons**

In `applyMetadata`, add `stopPlayback();` as the FIRST line of the function.

Replace the existing slider `input` listener:

```javascript
document.getElementById("time-slider").addEventListener("input", updateTimeLabel);
```

with:

```javascript
document.getElementById("time-slider").addEventListener("input", () => {
  if (playing) stopPlayback(); // user takes control
  updateTimeLabel();
});
```

(Programmatic `slider.value` assignment does not fire `input`, so the play loop never cancels itself.)

Add with the other listeners at the bottom:

```javascript
document.getElementById("play-btn").addEventListener("click", togglePlayback);
document.getElementById("step-back-btn").addEventListener("click", () => nudge(-1));
document.getElementById("step-fwd-btn").addEventListener("click", () => nudge(1));
```

- [ ] **Step 4: Syntax check and suite**

```bash
node --check static/app.js          # expect: exit 0
.venv/bin/python -m pytest          # expect: 72 passed
```

- [ ] **Step 5: Commit**

```bash
git add static/app.js
git commit -m "feat: play/pause and step controls for the time slider"
```

End the commit message with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 3: Browser verification (Playwright)

**Files:** none (verification only; fixes loop back into Tasks 1–2 files if found)

- [ ] **Step 1: Start a verification server on port 8765 (NOT 8000)**

```bash
[ -f data/demo_global.nc ] || .venv/bin/python scripts/make_demo_data.py
.venv/bin/python -c "
import uvicorn
from server.app import app
uvicorn.run(app, host='127.0.0.1', port=8765)
" > /tmp/ncse-anim.log 2>&1 &
sleep 3
```

- [ ] **Step 2: Drive the browser through the spec's checklist**

Navigate to http://127.0.0.1:8765/, open `/Users/ddamelin/Development/netCDF-segmenter-exporter/data/demo_global.nc` via the UI, then verify each (PASS/FAIL with evidence):

1. Nudge forward: time label advances to index 1's timestamp and the overlay URL changes (evaluate `overlay._url` before/after).
2. Nudge back at index 0: no-op (label unchanged). Nudge back from index 1 returns to index 0.
3. Play: button text becomes "⏸ Pause"; slider value increases on its own; overlay URL keeps changing. At Normal speed expect roughly 4 steps/sec (sample slider value ~2s apart; accept 5–10 steps).
4. Speed change mid-play to Fast: advance rate increases (sample again; accept anything clearly faster than before).
5. Pause: slider value freezes; button text returns to "▶ Play".
6. Manual slider drag during play (set value + dispatch `input` event) stops playback (button shows "▶ Play").
7. Play at the last step: jumps to index 0 and continues playing.
8. Play runs to the end (set slider near the end first, e.g. index 70 of 72, then play): stops at the last index with button back to "▶ Play".
9. Console: zero JS errors throughout.

- [ ] **Step 3: Teardown — MANDATORY**

```bash
lsof -ti:8765 | xargs kill 2>/dev/null; sleep 1; lsof -ti:8765 || echo clear
```

Close the browser. Never touch port 8000.

- [ ] **Step 4: No commit** (nothing changed); report results. Any FAIL loops back to a fix in the Task 1/2 files, then re-verify.
