# In-App File Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Browse… button that opens an in-sidebar file browser (server-side directory listings) so the user can pick a netCDF file by clicking instead of typing its absolute path. Spec: `docs/superpowers/specs/2026-06-11-file-picker-design.md`.

**Architecture:** New `server/browse.py` module (`list_directory`) + thin `GET /api/browse` endpoint in `server/app.py` mapping errors in the app's established style (404/400/403). Frontend: collapsible panel in the sidebar; clicking a file fills the existing path input and calls the existing `openFile()`. Last browsed directory remembered in localStorage.

**Tech Stack:** Python/FastAPI + pytest (backend, TDD); vanilla JS/HTML/CSS (frontend); Playwright browser pass for UI verification. Typed path entry is untouched.

**Constraint for all tasks:** the user's own server may be running on port 8000 — never start, stop, or use port 8000. Use port 8765 for live checks and kill it afterward. Current suite: 73 passed.

---

## File Structure

```
server/browse.py        # NEW: list_directory(path) -> {path, parent, dirs, files}
server/app.py           # + GET /api/browse (thin glue; error mapping)
static/index.html       # + Browse… button and #browse-panel markup
static/style.css        # + panel styles
static/app.js           # + panel logic (~70 lines)
tests/test_browse.py    # NEW: unit tests for list_directory
tests/test_api.py       # + 5 endpoint tests
README.md               # + one line under "Using the app"
```

---

### Task 1: `list_directory` in server/browse.py (TDD)

**Files:**
- Create: `server/browse.py`
- Create: `tests/test_browse.py`

- [ ] **Step 1: Write failing tests in `tests/test_browse.py`**

```python
import os
from pathlib import Path

import pytest

from server.browse import list_directory


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "beta").mkdir()
    (tmp_path / "Alpha").mkdir()
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / "Zebra.NC").write_bytes(b"x" * 10)
    (tmp_path / "apple.nc").write_bytes(b"x" * 5)
    (tmp_path / "data.nc4").write_bytes(b"x" * 7)
    (tmp_path / "notes.txt").write_text("not netcdf")
    (tmp_path / ".hidden.nc").write_bytes(b"x")
    return tmp_path


def test_lists_dirs_and_netcdf_files_sorted(tree):
    out = list_directory(tree)
    assert out["path"] == str(tree.resolve())
    assert [d["name"] for d in out["dirs"]] == ["Alpha", "beta"]
    assert [f["name"] for f in out["files"]] == ["apple.nc", "data.nc4", "Zebra.NC"]
    assert out["files"][0]["size_bytes"] == 5


def test_excludes_non_netcdf_and_dotfiles(tree):
    out = list_directory(tree)
    names = [f["name"] for f in out["files"]] + [d["name"] for d in out["dirs"]]
    assert "notes.txt" not in names
    assert ".hidden.nc" not in names
    assert ".hidden_dir" not in names


def test_parent_of_subdir_and_null_at_root(tree):
    out = list_directory(tree / "Alpha")
    assert out["parent"] == str(tree.resolve())
    root = list_directory("/")
    assert root["parent"] is None


def test_default_is_home():
    out = list_directory()
    assert out["path"] == str(Path.home())


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        list_directory("/nope/missing")


def test_file_path_raises(tree):
    with pytest.raises(NotADirectoryError):
        list_directory(tree / "apple.nc")


def test_broken_symlink_skipped(tree):
    os.symlink(tree / "gone.nc", tree / "broken.nc")
    out = list_directory(tree)
    assert "broken.nc" not in [f["name"] for f in out["files"]]
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/test_browse.py -v`**

Expected: FAIL — `ModuleNotFoundError: No module named 'server.browse'`. Observe before implementing.

- [ ] **Step 3: Write `server/browse.py`**

```python
"""List directories server-side so the frontend can offer a file picker.

Browsers never reveal a picked file's real path to web pages, so the
server (which runs on the user's machine) does the browsing instead.
"""

from pathlib import Path

NETCDF_SUFFIXES = {".nc", ".nc4"}


def list_directory(path=None):
    """Return {path, parent, dirs, files} for a directory.

    path=None lists the user's home directory. Dotfiles are hidden, files
    are filtered to netCDF suffixes, and entries whose metadata can't be
    read (broken symlinks, unreadable mounts) are skipped.

    Raises FileNotFoundError, NotADirectoryError, or PermissionError
    (e.g. macOS privacy-protected folders).
    """
    directory = Path(path).expanduser() if path else Path.home()
    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    dirs = []
    files = []
    for entry in directory.iterdir():  # raises PermissionError when blocked
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                dirs.append({"name": entry.name})
            elif entry.suffix.lower() in NETCDF_SUFFIXES and entry.is_file():
                files.append(
                    {"name": entry.name, "size_bytes": entry.stat().st_size}
                )
        except OSError:
            continue  # broken symlink or unreadable entry: skip it

    dirs.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda f: f["name"].lower())
    parent = None if directory.parent == directory else str(directory.parent)
    return {"path": str(directory), "parent": parent, "dirs": dirs, "files": files}
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_browse.py -v
```

Expected: `7 passed`. Full suite: `80 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/browse.py tests/test_browse.py
git commit -m "feat: server-side directory listing for the file picker"
```

End the commit message with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 2: `GET /api/browse` endpoint (TDD)

**Files:**
- Modify: `server/app.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Add failing tests to `tests/test_api.py`** (add `from pathlib import Path` to the imports at the top of the file)

```python
def test_browse_lists_directory(client, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.nc").write_bytes(b"xx")
    res = client.get("/api/browse", params={"path": str(tmp_path)})
    assert res.status_code == 200
    body = res.json()
    assert body["path"] == str(tmp_path.resolve())
    assert body["parent"] == str(tmp_path.resolve().parent)
    assert [d["name"] for d in body["dirs"]] == ["sub"]
    assert [f["name"] for f in body["files"]] == ["a.nc"]


def test_browse_defaults_to_home_and_needs_no_open_file(client):
    # note: no dataset is open in this fixture — browse must work anyway
    res = client.get("/api/browse")
    assert res.status_code == 200
    assert res.json()["path"] == str(Path.home())


def test_browse_missing_dir_returns_404(client):
    res = client.get("/api/browse", params={"path": "/nope/missing"})
    assert res.status_code == 404


def test_browse_file_path_returns_400(client, sample_nc):
    res = client.get("/api/browse", params={"path": str(sample_nc)})
    assert res.status_code == 400


def test_browse_unreadable_dir_returns_403(client, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        res = client.get("/api/browse", params={"path": str(locked)})
        assert res.status_code == 403
        assert "permission" in res.json()["detail"].lower()
    finally:
        locked.chmod(0o755)
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/test_api.py -v`**

Expected: the 5 new tests FAIL with 404 from the static-files mount (no `/api/browse` route — StaticFiles answers 404 for unknown paths, so missing-dir may "accidentally pass"; treat any test passing for the wrong reason as a failure to observe — confirm `test_browse_lists_directory` fails). Existing 23 tests pass.

- [ ] **Step 3: Add the endpoint to `server/app.py`**

Add to the imports block:

```python
from .browse import list_directory
```

Add the route after the `metadata` endpoint (it must be registered before the static mount at the bottom of the file, like every other route):

```python
@app.get("/api/browse")
async def browse(path: str | None = None):
    try:
        return list_directory(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Permission denied listing {path}. On macOS, folders like "
                "Downloads, Desktop, and Documents are privacy-protected: "
                "grant your terminal access in System Settings → Privacy & "
                "Security → Files & Folders, or browse elsewhere."
            ),
        )
```

- [ ] **Step 4: Run the full suite**

```bash
.venv/bin/python -m pytest
```

Expected: `85 passed`.

- [ ] **Step 5: Commit**

```bash
git add server/app.py tests/test_api.py
git commit -m "feat: /api/browse endpoint for the file picker"
```

End with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 3: Frontend panel (markup, styles, logic)

**Files:**
- Modify: `static/index.html` (File section)
- Modify: `static/style.css` (append)
- Modify: `static/app.js`
- Modify: `README.md`

- [ ] **Step 1: Update the File section in `static/index.html`**

Replace:

```html
      <input id="file-path" type="text" placeholder="/path/to/data.nc">
      <button id="open-btn">Open</button>
      <p id="file-info" class="muted"></p>
```

with:

```html
      <input id="file-path" type="text" placeholder="/path/to/data.nc">
      <div class="row">
        <button id="open-btn">Open</button>
        <button id="browse-btn" type="button">Browse&#8230;</button>
      </div>
      <div id="browse-panel" hidden>
        <div id="browse-header">
          <button id="browse-up-btn" type="button" aria-label="Parent folder">&#8593;</button>
          <span id="browse-path" title=""></span>
          <button id="browse-close-btn" type="button" aria-label="Close file browser">&#10005;</button>
        </div>
        <p id="browse-error" class="muted" hidden></p>
        <ul id="browse-list"></ul>
      </div>
      <p id="file-info" class="muted"></p>
```

- [ ] **Step 2: Append to `static/style.css`**

```css
#browse-panel {
  margin-top: 8px; border: 1px solid #ddd; border-radius: 4px; background: #fff;
}
#browse-header {
  display: flex; align-items: center; gap: 6px; padding: 6px;
  border-bottom: 1px solid #eee;
}
#browse-header button { margin-top: 0; padding: 2px 8px; }
#browse-path {
  flex: 1; font-size: 12px; color: #555;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#browse-list {
  list-style: none; margin: 0; padding: 4px; max-height: 40vh; overflow-y: auto;
}
#browse-list li {
  padding: 4px 6px; font-size: 13px; cursor: pointer; border-radius: 3px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#browse-list li:hover { background: #eef3ff; }
#browse-list li.empty { color: #888; cursor: default; }
#browse-list li.empty:hover { background: none; }
#browse-error { padding: 6px; margin: 0; }
```

- [ ] **Step 3: Add panel logic to `static/app.js`**

Add after the `buildFilters` function (module level):

```javascript
const BROWSE_DIR_KEY = "ncse:lastBrowseDir";
let browseParent = null;

function joinPath(dir, name) {
  return dir.endsWith("/") ? dir + name : `${dir}/${name}`;
}

async function loadBrowse(path) {
  const errEl = document.getElementById("browse-error");
  errEl.hidden = true;
  try {
    let url = "/api/browse";
    if (path) url += `?path=${encodeURIComponent(path)}`;
    const res = await fetch(url);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      errEl.textContent =
        typeof err.detail === "string" ? err.detail : res.statusText;
      errEl.hidden = false;
      return false; // previous listing stays visible beneath the error
    }
    renderBrowse(await res.json());
    return true;
  } catch (err) {
    errEl.textContent = `Request failed: ${err.message}`;
    errEl.hidden = false;
    return false;
  }
}

function renderBrowse(listing) {
  localStorage.setItem(BROWSE_DIR_KEY, listing.path);
  browseParent = listing.parent;
  const pathEl = document.getElementById("browse-path");
  pathEl.textContent = listing.path;
  pathEl.title = listing.path;
  document.getElementById("browse-up-btn").disabled = listing.parent === null;

  const list = document.getElementById("browse-list");
  list.innerHTML = "";
  for (const d of listing.dirs) {
    const li = document.createElement("li");
    li.textContent = `\u{1F4C1} ${d.name}`;
    li.addEventListener("click", () => loadBrowse(joinPath(listing.path, d.name)));
    list.appendChild(li);
  }
  for (const f of listing.files) {
    const li = document.createElement("li");
    li.textContent = `${f.name} (${(f.size_bytes / 1048576).toFixed(1)} MB)`;
    li.addEventListener("click", () => {
      document.getElementById("file-path").value = joinPath(listing.path, f.name);
      document.getElementById("browse-panel").hidden = true;
      openFile();
    });
    list.appendChild(li);
  }
  if (!listing.dirs.length && !listing.files.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "(no folders or netCDF files)";
    list.appendChild(li);
  }
}

async function openBrowsePanel() {
  document.getElementById("browse-panel").hidden = false;
  const list = document.getElementById("browse-list");
  if (!list.children.length) {
    list.innerHTML = '<li class="empty">Loading…</li>';
  }
  const remembered = localStorage.getItem(BROWSE_DIR_KEY);
  const ok = await loadBrowse(remembered || null);
  if (!ok && remembered) {
    await loadBrowse(null); // remembered dir vanished or is blocked: go home
  }
}
```

Add with the other listeners at the bottom of the file:

```javascript
document.getElementById("browse-btn").addEventListener("click", () => {
  const panel = document.getElementById("browse-panel");
  if (panel.hidden) openBrowsePanel();
  else panel.hidden = true;
});
document.getElementById("browse-up-btn").addEventListener("click", () => {
  if (browseParent) loadBrowse(browseParent);
});
document.getElementById("browse-close-btn").addEventListener("click", () => {
  document.getElementById("browse-panel").hidden = true;
});
```

- [ ] **Step 4: Add the README line**

In `README.md`, "Using the app" step 1, change:

```markdown
1. Enter a file path and click **Open** — variables, the time range, and the
   map extent populate from the file's metadata.
```

to:

```markdown
1. Enter a file path and click **Open**, or click **Browse…** to pick a
   netCDF file from an in-app folder listing — variables, the time range,
   and the map extent populate from the file's metadata.
```

- [ ] **Step 5: Verify**

```bash
node --check static/app.js          # expect: exit 0
.venv/bin/python -m pytest          # expect: 85 passed
```

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/style.css static/app.js README.md
git commit -m "feat: in-app file picker panel with browse, navigate, and pick"
```

End with the trailer line:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 4: Browser verification (Playwright)

**Files:** none (verification only; failures loop back into Task 1–3 files)

- [ ] **Step 1: Start a verification server on port 8765 (NOT 8000)**

```bash
[ -f data/demo_global.nc ] || .venv/bin/python scripts/make_demo_data.py
lsof -ti:8765 | xargs kill 2>/dev/null; sleep 1
.venv/bin/python -c "
import uvicorn
from server.app import app
uvicorn.run(app, host='127.0.0.1', port=8765)
" > /tmp/ncse-picker.log 2>&1 &
sleep 3
```

- [ ] **Step 2: Drive the browser; PASS/FAIL each with evidence**

Navigate to http://127.0.0.1:8765/ (clear `localStorage` first for a clean run: `localStorage.clear()`).

1. Click **Browse…** → panel opens showing the home directory (path header = home, list has folders).
2. Navigate: click into `Development`, then `netCDF-segmenter-exporter`, then `data` → list shows `demo_global.nc (9.0 MB)` and any other .nc files; no dotfiles anywhere along the way.
3. Up button: click **↑** → back in the project directory listing. At `/` (navigate up repeatedly) the ↑ button is disabled.
4. Pick the file: navigate back into `data`, click `demo_global.nc` → panel closes, `#file-path` contains the absolute path, the file OPENS (variable select populates, overlay appears).
5. Memory: reload the page, click **Browse…** → panel starts in the `data` directory (localStorage), not home.
6. Toggle/close: **Browse…** toggles the panel; **✕** closes it.
7. Error inline: evaluate `loadBrowse('/nope/missing')` → error text appears inside the panel AND the previous listing is still visible/clickable beneath it.
8. Console: zero JS errors throughout.

- [ ] **Step 3: Teardown — MANDATORY**

```bash
lsof -ti:8765 | xargs kill 2>/dev/null; sleep 1; lsof -ti:8765 || echo clear
```

Close the browser. Never touch port 8000.

- [ ] **Step 4: No commit** (nothing changed); report results. Any FAIL loops back to a fix, then re-verify.
