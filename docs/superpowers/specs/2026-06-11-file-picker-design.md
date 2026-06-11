# In-App File Picker — Design

**Date:** 2026-06-11
**Status:** Approved
**Scope:** New `server/browse.py` module + one endpoint in `server/app.py`;
frontend panel in `static/index.html` / `static/style.css` / `static/app.js`;
tests; one README line. Typed path entry stays unchanged — the picker is
additive.

## Purpose

Pick the netCDF file by browsing instead of typing an absolute path. Browsers
never reveal a picked file's real path to web pages (and uploading the
contents would copy gigabytes and defeat lazy loading), so the server — which
runs on the same machine — does the browsing and returns listings.

## Backend

### `server/browse.py` — `list_directory(path=None) -> dict`

- `path=None` → `Path.home()`.
- Returns:

```json
{
  "path": "/abs/resolved/dir",
  "parent": "/abs/parent or null when at filesystem root",
  "dirs":  [{"name": "subdir"}, ...],
  "files": [{"name": "data.nc", "size_bytes": 123}, ...]
}
```

- `files` includes only netCDF extensions: `.nc`, `.nc4` (case-insensitive).
- Entries starting with `.` (dotfiles) are hidden. No show-hidden toggle
  (YAGNI).
- `dirs` and `files` each sorted alphabetically, case-insensitive.
- Symlinked directories are listed and navigable (default `is_dir()`
  follows). Entries whose `stat` fails (broken symlinks, unreadable items)
  are skipped silently rather than failing the listing.
- Raises: `FileNotFoundError` (missing path), `NotADirectoryError` (file
  path given), `PermissionError` (e.g. macOS TCC-protected folder).

### `GET /api/browse?path=<dir>` in `server/app.py`

Thin glue over `list_directory`, mapping errors in the app's established
style:

- `FileNotFoundError` → 404
- `NotADirectoryError` → 400
- `PermissionError` → 403 with the same macOS Files-&-Folders guidance text
  pattern `/api/open` uses

No path restrictions: localhost single-user tool, and `/api/open` already
accepts arbitrary paths — same trust model, no new security surface. Works
with no dataset open (independent of `_require_open`).

## Frontend

- A **Browse…** button (`#browse-btn`) next to the existing path input in the
  File section.
- Clicking toggles an inline collapsible panel (`#browse-panel`, hidden by
  default) inside the sidebar under the File section — not a modal, so no
  z-index interaction with Leaflet.
- Panel contents:
  - Header row: current folder path (`#browse-path`, CSS-truncated with
    `title` attr for the full path), an **↑** up button (`#browse-up-btn`,
    disabled — not hidden — when `parent` is null), and a **Close** button
    (`#browse-close-btn`).
  - Scrollable list (`#browse-list`, max-height ≈ 40vh): folders first
    (📁 prefix), then files with size in MB to one decimal place (matching
    the existing file-info display). Click a folder →
    navigate into it; click a file → fill `#file-path` with the absolute
    path, call the existing `openFile()`, close the panel.
- **Last-directory memory:** localStorage key `ncse:lastBrowseDir`, written
  on every successful listing. Opening the panel starts there; if listing it
  fails (deleted, blocked), fall back to home (no `path` param).
- **Errors inline:** a message area (`#browse-error`) inside the panel shows
  the response's `detail` (e.g. the macOS permission guidance) while the
  panel stays usable — user can go up or elsewhere.
- Brief "Loading…" text in the list area during fetches.
- Absolute path of a clicked file = `path + "/" + name` (the API's `path` is
  resolved-absolute; root "/" join must not produce "//" — guard it).

## Error handling

All failures surface either inline in the panel (`#browse-error`, for browse
calls) or via the existing `status` line (for the subsequent `/api/open`,
unchanged).

## Testing

- **pytest, `tests/test_browse.py`** (unit, tmp_path trees): extension
  filtering, dotfile hiding, case-insensitive sorting, parent null at root,
  default-to-home, FileNotFoundError/NotADirectoryError raised, skipping
  broken symlinks.
- **pytest, `tests/test_api.py`**: 200 shape, 404 missing, 400 non-dir,
  403 unreadable dir (chmod 000), works with no dataset open.
- **Playwright pass:** open panel (starts at home), navigate to the project's
  `data/` dir, click the demo file → it opens (variables populate); up
  button; close button; localStorage remembered across panel reopen.
  Permission-denied UI message is covered by the API test, not the browser
  pass (can't simulate TCC in CI).
- README: one line documenting the Browse button under "Using the app".
