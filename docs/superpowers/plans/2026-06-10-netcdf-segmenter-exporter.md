# netCDF Segmenter & Exporter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app that opens one (possibly large) gridded netCDF file, shows a time slice of a chosen variable on a zoomable world map, lets the user filter by drawn bounding box or polygon, time range, and variable value range, and exports the subset as netCDF or CSV.

**Architecture:** Python backend (FastAPI) uses xarray with lazy loading so only metadata and the requested time slice are read from large files. The backend renders slices to colormapped PNGs and applies all filters server-side. A no-build-step frontend (vanilla JS + Leaflet + leaflet-draw, loaded from CDN) displays the map, draw tools, and filter controls, and downloads exports via the API.

**Tech Stack:** Python 3.10+, xarray, netCDF4, numpy, pandas, matplotlib (rendering + polygon point-in-path test), FastAPI, uvicorn, pytest, httpx (test client). Frontend: Leaflet 1.9.4, leaflet-draw 1.0.4, vanilla JS/CSS.

**Out of scope (YAGNI, noted so nobody "helpfully" adds them):** multiple files at once, shapes crossing the antimeridian, variables with extra dimensions beyond (time, lat, lon) such as vertical levels (the slice endpoint returns 400 for those), authentication, streaming/chunked exports, colormap selection.

---

## File Structure

```
netCDF-segmenter-exporter/
├── pyproject.toml              # package metadata, deps, pytest config
├── .gitignore
├── README.md                   # Task 15
├── server/
│   ├── __init__.py
│   ├── __main__.py             # entrypoint: python -m server [file.nc]
│   ├── app.py                  # FastAPI routes + module-level DatasetManager
│   ├── dataset.py              # coord detection, normalization, DatasetManager
│   ├── rendering.py            # time slice -> colormapped PNG bytes
│   ├── subset.py               # bbox / polygon / time / value filters
│   └── export.py               # subset -> netCDF bytes / DataFrame / CSV bytes
├── static/
│   ├── index.html              # single page UI
│   ├── style.css
│   └── app.js                  # map, draw tools, controls, export
├── scripts/
│   └── make_demo_data.py       # generates data/demo_global.nc for manual testing
├── data/                       # gitignored; demo + user data lives here
└── tests/
    ├── conftest.py             # synthetic netCDF fixtures
    ├── test_fixtures.py
    ├── test_dataset.py
    ├── test_rendering.py
    ├── test_subset.py
    ├── test_export.py
    └── test_api.py
```

**Responsibilities:** `dataset.py` owns everything about understanding a file (which coords are lat/lon/time, normalizing them, metadata). `subset.py` owns turning filter parameters into a smaller `xr.Dataset`. `rendering.py` and `export.py` each consume a Dataset and produce bytes. `app.py` is thin glue: HTTP in, the above modules out.

**Data contracts used throughout (defined once here, referenced by name):**

- `coords` dict: `{"lat": "<coord name>", "lon": "<coord name>", "time": "<coord name or None>"}` — produced by `detect_coords`, consumed by every other module.
- `bbox` dict: `{"west": float, "south": float, "east": float, "north": float}`.
- `polygon`: list of `[lon, lat]` float pairs forming a ring (GeoJSON coordinate order; first ring of a leaflet-draw polygon).
- `time_range` dict: `{"start": ISO-8601 string or null, "end": ISO-8601 string or null}`.
- `var_filter` dict: `{"variable": str, "min": float or null, "max": float or null}` — cells failing the condition are masked to NaN in **all** variables (dropped entirely from CSV rows).

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `server/__init__.py`
- Create: `static/index.html` (placeholder; replaced in Task 13)

- [ ] **Step 1: Initialize the git repository with `main` as the default branch**

```bash
cd path/to/netcdf-segmenter-exporter
git init -b main
```

Expected: `Initialized empty Git repository in .../netCDF-segmenter-exporter/.git/`

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "netcdf-segmenter-exporter"
version = "0.1.0"
description = "Visualize, segment, and export spatio-temporal netCDF data"
requires-python = ">=3.10"
dependencies = [
    "xarray>=2023.1",
    "netCDF4>=1.6",
    "numpy>=1.24",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "fastapi>=0.110",
    "uvicorn>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
]

[tool.setuptools]
packages = ["server"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/
.DS_Store
```

- [ ] **Step 4: Create `server/__init__.py`** (empty file)

```python
```

- [ ] **Step 5: Create placeholder `static/index.html`**

The real UI arrives in Task 13, but `app.py` (Task 11) mounts this directory at import time and `StaticFiles` requires it to exist, so create it now:

```html
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>netCDF Segmenter & Exporter</title></head>
<body><p>UI coming soon.</p></body>
</html>
```

- [ ] **Step 6: Create the virtualenv and install dependencies**

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Expected: ends with `Successfully installed ... netcdf-segmenter-exporter-0.1.0 ...` (netCDF4 and matplotlib ship binary wheels for macOS; no compiler needed).

- [ ] **Step 7: Verify pytest runs (no tests yet)**

```bash
.venv/bin/python -m pytest
```

Expected: `no tests ran` with exit code 5 — that is fine at this stage.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore server/__init__.py static/index.html
git commit -m "chore: scaffold project with pyproject, venv deps, and placeholder static page"
```

---

### Task 2: Synthetic netCDF test fixtures

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_fixtures.py`

Real netCDF files are too big to commit, so every test builds tiny synthetic ones. Two fixtures: a "well-behaved" file (ascending lats, −180..180 lons) and a "rotated" file (descending lats, 0..360 lons — extremely common in climate model output) to prove normalization works.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _build_dataset(lats, lons, times):
    rng = np.random.default_rng(42)
    shape = (len(times), len(lats), len(lons))
    temperature = 15 + 10 * rng.random(shape)   # uniform in [15, 25)
    humidity = 100 * rng.random(shape)          # uniform in [0, 100)
    return xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lat", "lon"),
                temperature,
                {"units": "degC", "long_name": "Air Temperature"},
            ),
            "humidity": (
                ("time", "lat", "lon"),
                humidity,
                {"units": "%", "long_name": "Relative Humidity"},
            ),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
    )


@pytest.fixture
def sample_nc(tmp_path):
    """Well-behaved file: ascending lat, -180..180 lon, 4 daily steps."""
    lats = np.arange(-85.0, 86.0, 10.0)      # 18 cells
    lons = np.arange(-175.0, 176.0, 10.0)    # 36 cells
    times = pd.date_range("2020-01-01", periods=4, freq="D")
    ds = _build_dataset(lats, lons, times)
    path = tmp_path / "sample.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


@pytest.fixture
def rotated_nc(tmp_path):
    """0..360 longitudes and descending latitudes, as in many model outputs."""
    lats = np.arange(85.0, -86.0, -10.0)     # 18 cells, descending
    lons = np.arange(5.0, 356.0, 10.0)       # 36 cells, 0..360 convention
    times = pd.date_range("2020-01-01", periods=2, freq="D")
    ds = _build_dataset(lats, lons, times)
    path = tmp_path / "rotated.nc"
    ds.to_netcdf(path)
    ds.close()
    return path
```

- [ ] **Step 2: Write `tests/test_fixtures.py`**

```python
import xarray as xr


def test_sample_fixture_shape(sample_nc):
    with xr.open_dataset(sample_nc) as ds:
        assert ds.sizes == {"time": 4, "lat": 18, "lon": 36}
        assert set(ds.data_vars) == {"temperature", "humidity"}


def test_rotated_fixture_conventions(rotated_nc):
    with xr.open_dataset(rotated_nc) as ds:
        assert float(ds["lat"][0]) > float(ds["lat"][-1])   # descending
        assert float(ds["lon"].max()) > 180                 # 0..360
```

- [ ] **Step 3: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_fixtures.py -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_fixtures.py
git commit -m "test: add synthetic netCDF fixtures (standard and rotated conventions)"
```

---

### Task 3: Coordinate detection

**Files:**
- Create: `server/dataset.py`
- Create: `tests/test_dataset.py`

Real files name their coordinates `lat`/`latitude`/`y` etc. `detect_coords` finds them by name, CF `standard_name`, or units, and returns the `coords` dict contract.

- [ ] **Step 1: Write the failing tests in `tests/test_dataset.py`**

```python
import numpy as np
import pytest
import xarray as xr

from server.dataset import detect_coords


def test_detect_coords_standard_names(sample_nc):
    with xr.open_dataset(sample_nc) as ds:
        assert detect_coords(ds) == {"lat": "lat", "lon": "lon", "time": "time"}


def test_detect_coords_by_long_names_and_units():
    ds = xr.Dataset(
        {"v": (("latitude", "longitude"), np.zeros((2, 2)))},
        coords={
            "latitude": ("latitude", [0.0, 1.0], {"units": "degrees_north"}),
            "longitude": ("longitude", [0.0, 1.0], {"units": "degrees_east"}),
        },
    )
    assert detect_coords(ds) == {"lat": "latitude", "lon": "longitude", "time": None}


def test_detect_coords_raises_without_latlon():
    ds = xr.Dataset(
        {"v": (("a", "b"), np.zeros((2, 2)))},
        coords={"a": [0, 1], "b": [0, 1]},
    )
    with pytest.raises(ValueError):
        detect_coords(ds)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_dataset.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'server.dataset'` (or ImportError).

- [ ] **Step 3: Write `server/dataset.py`**

```python
"""Open netCDF files lazily and understand their coordinate layout."""

import numpy as np
import xarray as xr

LAT_NAMES = {"lat", "latitude", "y"}
LON_NAMES = {"lon", "longitude", "x"}
TIME_NAMES = {"time", "t"}


def _match_coord(ds, names, units_substrings):
    for name, coord in ds.coords.items():
        lname = str(name).lower()
        std = str(coord.attrs.get("standard_name", "")).lower()
        units = str(coord.attrs.get("units", "")).lower()
        if lname in names or std in names or any(u in units for u in units_substrings):
            return str(name)
    return None


def detect_coords(ds):
    """Return {"lat": name, "lon": name, "time": name-or-None} for a dataset.

    Raises ValueError if latitude or longitude cannot be identified.
    """
    lat = _match_coord(ds, LAT_NAMES, ("degrees_north", "degree_north"))
    lon = _match_coord(ds, LON_NAMES, ("degrees_east", "degree_east"))
    time = None
    for name, coord in ds.coords.items():
        if str(name).lower() in TIME_NAMES or np.issubdtype(coord.dtype, np.datetime64):
            time = str(name)
            break
    if lat is None or lon is None:
        raise ValueError(
            "Could not identify latitude/longitude coordinates in this file"
        )
    return {"lat": lat, "lon": lon, "time": time}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_dataset.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add server/dataset.py tests/test_dataset.py
git commit -m "feat: detect lat/lon/time coordinates by name, standard_name, or units"
```

---

### Task 4: Coordinate normalization

**Files:**
- Modify: `server/dataset.py`
- Modify: `tests/test_dataset.py`

Maps and `.sel(slice(...))` both need ascending coordinates in −180..180 longitude. Normalize once at open time so every downstream module can assume it.

- [ ] **Step 1: Add failing tests to `tests/test_dataset.py`**

Append (and add `normalize_coords` to the existing import from `server.dataset`):

```python
from server.dataset import detect_coords, normalize_coords  # update existing import


def test_normalize_sorts_lat_ascending_and_wraps_lon(rotated_nc):
    with xr.open_dataset(rotated_nc) as ds:
        coords = detect_coords(ds)
        out = normalize_coords(ds, coords)
        lats = out["lat"].values
        lons = out["lon"].values
        assert (np.diff(lats) > 0).all()
        assert (np.diff(lons) > 0).all()
        assert lons.min() >= -180.0 and lons.max() <= 180.0


def test_normalize_preserves_values(rotated_nc):
    with xr.open_dataset(rotated_nc) as orig:
        coords = detect_coords(orig)
        out = normalize_coords(orig, coords)
        # lon 355 in the 0..360 file becomes lon -5 after normalization
        expected = float(orig["temperature"].isel(time=0).sel(lat=85.0, lon=355.0))
        actual = float(out["temperature"].isel(time=0).sel(lat=85.0, lon=-5.0))
        assert actual == expected


def test_normalize_leaves_wellbehaved_file_alone(sample_nc):
    with xr.open_dataset(sample_nc) as ds:
        coords = detect_coords(ds)
        out = normalize_coords(ds, coords)
        assert (out["lat"].values == ds["lat"].values).all()
        assert (out["lon"].values == ds["lon"].values).all()
```

- [ ] **Step 2: Run tests to verify the new ones fail**

```bash
.venv/bin/python -m pytest tests/test_dataset.py -v
```

Expected: 3 prior tests pass; new tests FAIL with `ImportError: cannot import name 'normalize_coords'`.

- [ ] **Step 3: Add `normalize_coords` to `server/dataset.py`**

```python
def normalize_coords(ds, coords):
    """Return a view of ds with ascending latitude and longitude in [-180, 180].

    xarray's sortby/assign_coords operate on the (small) coordinate arrays and
    keep data variables lazy, so this is cheap even for huge files.
    """
    lat, lon = coords["lat"], coords["lon"]
    if float(ds[lat][0]) > float(ds[lat][-1]):
        ds = ds.sortby(lat)
    lon_vals = ds[lon].values
    if float(lon_vals.max()) > 180.0:
        ds = ds.assign_coords({lon: ((lon_vals + 180.0) % 360.0) - 180.0})
        ds = ds.sortby(lon)
    return ds
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_dataset.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add server/dataset.py tests/test_dataset.py
git commit -m "feat: normalize latitudes to ascending and longitudes to -180..180"
```

---

### Task 5: DatasetManager — lazy open + metadata

**Files:**
- Modify: `server/dataset.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/conftest.py`

One open file at a time. `xr.open_dataset` is lazy by default — data is only read when a slice/filter actually touches it — so "load only a portion of large files" falls out for free. Metadata reads only coordinate arrays and attributes.

- [ ] **Step 1: Add failing tests to `tests/test_dataset.py`**

Append (and extend the import to include `DatasetManager`):

```python
from server.dataset import DatasetManager, detect_coords, normalize_coords  # update import


def test_manager_open_missing_file_raises():
    m = DatasetManager()
    with pytest.raises(FileNotFoundError):
        m.open("/nope/missing.nc")


def test_manager_open_non_netcdf_raises_value_error(tmp_path):
    bad = tmp_path / "bad.nc"
    bad.write_text("this is not a netCDF file")
    m = DatasetManager()
    with pytest.raises(ValueError):
        m.open(bad)


def test_manager_metadata(sample_nc):
    m = DatasetManager()
    meta = m.open(sample_nc)
    names = [v["name"] for v in meta["variables"]]
    assert names == ["temperature", "humidity"]
    assert meta["variables"][0]["units"] == "degC"
    assert meta["time"]["start"].startswith("2020-01-01")
    assert meta["time"]["end"].startswith("2020-01-04")
    assert meta["time"]["count"] == 4
    assert len(meta["time"]["values"]) == 4
    assert meta["extent"] == {
        "south": -85.0, "north": 85.0, "west": -175.0, "east": 175.0,
    }
    assert meta["size_bytes"] > 0
    m.close()


def test_manager_open_replaces_previous_file(sample_nc, rotated_nc):
    m = DatasetManager()
    m.open(sample_nc)
    meta = m.open(rotated_nc)
    assert meta["time"]["count"] == 2
    m.close()
```

- [ ] **Step 2: Run tests to verify the new ones fail**

```bash
.venv/bin/python -m pytest tests/test_dataset.py -v
```

Expected: ImportError on `DatasetManager`; 6 earlier tests pass.

- [ ] **Step 3: Add `DatasetManager` to `server/dataset.py`**

Add to the imports at the top:

```python
from pathlib import Path

import pandas as pd
```

Then append:

```python
def _iso(value):
    """Render a time coordinate value as an ISO-8601 string."""
    try:
        return pd.Timestamp(value).isoformat()
    except (ValueError, TypeError):
        return str(value)  # cftime / non-standard calendars


class DatasetManager:
    """Holds the single currently-open dataset, lazily loaded."""

    def __init__(self):
        self.ds = None
        self.coords = None
        self.path = None

    def open(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            ds = xr.open_dataset(path)  # lazy: no data variables are read yet
        except Exception as exc:
            raise ValueError(f"Could not open {path} as netCDF: {exc}") from exc
        try:
            coords = detect_coords(ds)
            ds = normalize_coords(ds, coords)
        except ValueError:
            ds.close()
            raise
        self.close()
        self.ds, self.coords, self.path = ds, coords, path
        return self.metadata()

    def close(self):
        if self.ds is not None:
            self.ds.close()
        self.ds = self.coords = self.path = None

    def metadata(self):
        if self.ds is None:
            raise RuntimeError("No dataset is open")
        lat, lon, time = self.coords["lat"], self.coords["lon"], self.coords["time"]

        variables = []
        for name, da in self.ds.data_vars.items():
            if lat in da.dims and lon in da.dims:  # only georeferenced variables
                variables.append({
                    "name": str(name),
                    "long_name": str(da.attrs.get("long_name", name)),
                    "units": str(da.attrs.get("units", "")),
                    "dims": [str(d) for d in da.dims],
                    "shape": [int(s) for s in da.shape],
                })

        time_info = None
        if time is not None:
            tvals = self.ds[time].values
            time_info = {
                "start": _iso(tvals[0]),
                "end": _iso(tvals[-1]),
                "count": int(tvals.size),
                # full list lets the UI label its slider; omit when huge
                "values": [_iso(t) for t in tvals] if tvals.size <= 2000 else None,
            }

        return {
            "path": str(self.path),
            "size_bytes": self.path.stat().st_size,
            "variables": variables,
            "time": time_info,
            "extent": {
                "south": float(self.ds[lat].min()),
                "north": float(self.ds[lat].max()),
                "west": float(self.ds[lon].min()),
                "east": float(self.ds[lon].max()),
            },
        }
```

- [ ] **Step 4: Add a shared `open_sample` fixture to `tests/conftest.py`**

Tasks 6–10 all need an opened, normalized dataset. Append to `tests/conftest.py`:

```python
from server.dataset import DatasetManager


@pytest.fixture
def open_sample(sample_nc):
    """(ds, coords) for the opened, normalized sample file."""
    m = DatasetManager()
    m.open(sample_nc)
    yield m.ds, m.coords
    m.close()
```

- [ ] **Step 5: Run the whole suite**

```bash
.venv/bin/python -m pytest -v
```

Expected: `12 passed` (2 fixtures + 10 dataset)

- [ ] **Step 6: Commit**

```bash
git add server/dataset.py tests/test_dataset.py tests/conftest.py
git commit -m "feat: DatasetManager with lazy open, metadata, and single-file lifecycle"
```

---

### Task 6: Render a time slice to PNG

**Files:**
- Create: `server/rendering.py`
- Create: `tests/test_rendering.py`

The map shows the slice as a Leaflet `imageOverlay`, so the backend produces a borderless colormapped PNG. `matplotlib.image.imsave` writes the array pixel-for-pixel (no axes/figure machinery) and renders NaN cells as transparent via the colormap's "bad" color.

- [ ] **Step 1: Write failing tests in `tests/test_rendering.py`**

```python
import numpy as np
import xarray as xr

from server.rendering import render_slice_png

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_render_returns_png_and_value_range(open_sample):
    ds, coords = open_sample
    png, vmin, vmax = render_slice_png(ds, coords, "temperature", time_index=0)
    assert png[:8] == PNG_MAGIC
    assert vmin < vmax
    assert 15.0 <= vmin <= vmax <= 25.0  # fixture temperature range


def test_render_handles_nan_cells():
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.array([[1.0, np.nan], [3.0, 4.0]]))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    coords = {"lat": "lat", "lon": "lon", "time": None}
    png, vmin, vmax = render_slice_png(ds, coords, "v")
    assert png[:8] == PNG_MAGIC
    assert (vmin, vmax) == (1.0, 4.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_rendering.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'server.rendering'`.

- [ ] **Step 3: Write `server/rendering.py`**

```python
"""Render one time slice of one variable as a colormapped PNG."""

import io

import numpy as np
from matplotlib import image as mpimage


def render_slice_png(ds, coords, variable, time_index=0, cmap="viridis"):
    """Return (png_bytes, vmin, vmax) for variable at time_index.

    Only the requested slice is read from disk (xarray lazy indexing).
    NaN cells render transparent. Raises ValueError if the slice is not 2D
    (e.g. the variable has a vertical-level dimension).
    """
    da = ds[variable]
    tname = coords.get("time")
    if tname and tname in da.dims:
        da = da.isel({tname: int(time_index)})
    if da.ndim != 2:
        raise ValueError(
            f"Variable {variable!r} is not a 2D lat/lon slice (dims: {da.dims})"
        )
    da = da.transpose(coords["lat"], coords["lon"])
    data = np.asarray(da.values, dtype=float)

    vmin = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
    vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0  # avoid a degenerate color scale

    buf = io.BytesIO()
    # origin="lower": row 0 is the southernmost latitude (coords are ascending)
    mpimage.imsave(buf, data, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", format="png")
    return buf.getvalue(), vmin, vmax
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_rendering.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add server/rendering.py tests/test_rendering.py
git commit -m "feat: render time slices as colormapped PNGs with transparent NaN cells"
```

---

### Task 7: Subset filters — time range and bounding box

**Files:**
- Create: `server/subset.py`
- Create: `tests/test_subset.py`

`apply_filters(ds, coords, bbox=None, polygon=None, time_range=None, var_filter=None)` is the single entry point for all filtering; this task implements the two cheap index-based filters. Order matters for performance: select (crop) before any masking so `.where()` only ever touches the cropped region.

- [ ] **Step 1: Write failing tests in `tests/test_subset.py`**

```python
import numpy as np

from server.subset import apply_filters


def test_time_range_filter(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords, time_range={"start": "2020-01-02", "end": "2020-01-03"}
    )
    assert out.sizes["time"] == 2


def test_time_range_open_ended(open_sample):
    ds, coords = open_sample
    out = apply_filters(ds, coords, time_range={"start": "2020-01-03", "end": None})
    assert out.sizes["time"] == 2  # Jan 3 and Jan 4


def test_bbox_filter(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords, bbox={"west": -20.0, "south": -10.0, "east": 30.0, "north": 40.0}
    )
    # lat cells at -5,5,15,25,35 and lon cells at -15,-5,5,15,25
    assert out.sizes["lat"] == 5
    assert out.sizes["lon"] == 5
    assert float(out["lat"].min()) >= -10.0
    assert float(out["lat"].max()) <= 40.0
    assert float(out["lon"].min()) >= -20.0
    assert float(out["lon"].max()) <= 30.0


def test_no_filters_returns_full_dataset(open_sample):
    ds, coords = open_sample
    out = apply_filters(ds, coords)
    assert out.sizes == ds.sizes
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_subset.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'server.subset'`.

- [ ] **Step 3: Write `server/subset.py`**

```python
"""Filter a dataset by time range, bounding box, polygon, and variable values."""

import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath


def apply_filters(ds, coords, bbox=None, polygon=None, time_range=None, var_filter=None):
    """Return a subset of ds. Assumes coords are normalized (ascending, -180..180).

    Cheap index-based selections (time, bbox, polygon bounding box) run first
    so the masking steps only touch the cropped region.
    """
    lat, lon, time = coords["lat"], coords["lon"], coords.get("time")
    out = ds

    if time_range and time:
        out = out.sel(
            {time: slice(time_range.get("start"), time_range.get("end"))}
        )

    if bbox:
        out = out.sel({
            lat: slice(bbox["south"], bbox["north"]),
            lon: slice(bbox["west"], bbox["east"]),
        })

    if polygon:
        out = _apply_polygon(out, lat, lon, polygon)

    if var_filter:
        out = _apply_var_filter(out, var_filter)

    return out


def _apply_polygon(out, lat, lon, polygon):
    raise NotImplementedError  # implemented in the next task (TDD)


def _apply_var_filter(out, var_filter):
    raise NotImplementedError  # implemented two tasks from now (TDD)
```

(The two stubs keep `apply_filters`'s full signature honest — passing `polygon` or `var_filter` fails loudly instead of being silently ignored — and Tasks 8 and 9 replace them test-first.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_subset.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add server/subset.py tests/test_subset.py
git commit -m "feat: subset filtering by time range and bounding box"
```

---

### Task 8: Subset filters — polygon mask

**Files:**
- Modify: `server/subset.py` (replace the `_apply_polygon` stub)
- Modify: `tests/test_subset.py`

The frontend sends the drawn polygon as a ring of `[lon, lat]` pairs. Cells outside the ring become NaN (so netCDF export keeps the grid shape with masked values, and CSV export drops those rows).

- [ ] **Step 1: Add failing tests to `tests/test_subset.py`**

```python
def test_polygon_filter_crops_and_masks(open_sample):
    ds, coords = open_sample
    # triangle: base from (-60,-30) to (60,-30), apex at (0,60); [lon, lat] order
    polygon = [[-60.0, -30.0], [60.0, -30.0], [0.0, 60.0], [-60.0, -30.0]]
    out = apply_filters(ds, coords, polygon=polygon)

    # cropped to the triangle's bounding box
    assert float(out["lat"].min()) >= -30.0
    assert float(out["lat"].max()) <= 60.0
    assert float(out["lon"].min()) >= -60.0
    assert float(out["lon"].max()) <= 60.0

    # (lon=5, lat=5) is inside the triangle; (lon=-55, lat=55) is in the
    # bounding box but outside the triangle, so it must be masked
    inside = out["temperature"].isel(time=0).sel(lat=5.0, lon=5.0)
    outside = out["temperature"].isel(time=0).sel(lat=55.0, lon=-55.0)
    assert not np.isnan(float(inside))
    assert np.isnan(float(outside))


def test_polygon_mask_applies_to_all_variables(open_sample):
    ds, coords = open_sample
    polygon = [[-60.0, -30.0], [60.0, -30.0], [0.0, 60.0], [-60.0, -30.0]]
    out = apply_filters(ds, coords, polygon=polygon)
    t_nan = np.isnan(out["temperature"].values)
    h_nan = np.isnan(out["humidity"].values)
    assert np.array_equal(t_nan, h_nan)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_subset.py -v
```

Expected: the 2 new tests FAIL with `NotImplementedError`; the 4 earlier tests pass.

- [ ] **Step 3: Replace the `_apply_polygon` stub in `server/subset.py`**

```python
def _apply_polygon(out, lat, lon, polygon):
    pts = np.asarray(polygon, dtype=float)  # rows of [lon, lat]
    # crop to the polygon's bounding box first, then mask cells outside the ring
    out = out.sel({
        lat: slice(float(pts[:, 1].min()), float(pts[:, 1].max())),
        lon: slice(float(pts[:, 0].min()), float(pts[:, 0].max())),
    })
    lon2d, lat2d = np.meshgrid(out[lon].values, out[lat].values)
    inside = MplPath(pts).contains_points(
        np.column_stack([lon2d.ravel(), lat2d.ravel()])
    )
    mask = xr.DataArray(
        inside.reshape(lat2d.shape),
        dims=(lat, lon),
        coords={lat: out[lat], lon: out[lon]},
    )
    return out.where(mask)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_subset.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add server/subset.py tests/test_subset.py
git commit -m "feat: polygon crop-and-mask filtering"
```

---

### Task 9: Subset filters — variable value range

**Files:**
- Modify: `server/subset.py` (replace the `_apply_var_filter` stub)
- Modify: `tests/test_subset.py`

"Filter by a variable" = keep only cells where the chosen variable falls within [min, max]; everything else is masked across **all** variables so exports stay consistent row-wise.

- [ ] **Step 1: Add failing tests to `tests/test_subset.py`**

```python
import pytest  # add to imports at top of file if not present


def test_var_filter_min_only(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords, var_filter={"variable": "temperature", "min": 20.0, "max": None}
    )
    temp = out["temperature"].values
    assert np.nanmin(temp) >= 20.0
    assert np.isnan(temp).any()  # fixture spans 15..25, so some cells were masked


def test_var_filter_masks_all_variables(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords, var_filter={"variable": "temperature", "min": 20.0, "max": 24.0}
    )
    assert np.array_equal(
        np.isnan(out["temperature"].values), np.isnan(out["humidity"].values)
    )


def test_var_filter_unknown_variable_raises(open_sample):
    ds, coords = open_sample
    with pytest.raises(ValueError):
        apply_filters(ds, coords, var_filter={"variable": "nope", "min": 0, "max": 1})


def test_filters_compose(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords,
        bbox={"west": -20.0, "south": -10.0, "east": 30.0, "north": 40.0},
        time_range={"start": "2020-01-01", "end": "2020-01-02"},
        var_filter={"variable": "temperature", "min": 20.0, "max": None},
    )
    assert out.sizes == {"time": 2, "lat": 5, "lon": 5}
    assert np.nanmin(out["temperature"].values) >= 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_subset.py -v
```

Expected: the 4 new tests FAIL with `NotImplementedError`; the 6 earlier tests pass. (`test_var_filter_unknown_variable_raises` fails because `NotImplementedError` is not a `ValueError`.)

- [ ] **Step 3: Replace the `_apply_var_filter` stub in `server/subset.py`**

```python
def _apply_var_filter(out, var_filter):
    name = var_filter["variable"]
    if name not in out.data_vars:
        raise ValueError(f"Unknown filter variable: {name}")
    da = out[name]
    cond = xr.ones_like(da, dtype=bool)
    if var_filter.get("min") is not None:
        cond = cond & (da >= var_filter["min"])
    if var_filter.get("max") is not None:
        cond = cond & (da <= var_filter["max"])
    return out.where(cond)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_subset.py -v
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add server/subset.py tests/test_subset.py
git commit -m "feat: variable value-range filtering"
```

---

### Task 10: Export — netCDF bytes and CSV

**Files:**
- Create: `server/export.py`
- Create: `tests/test_export.py`

CSV contract from the spec: a `latitude` column, a `longitude` column, and a column per variable (plus `time` when the file has a time axis). Rows where every variable is NaN (i.e. masked out by polygon/value filters) are dropped.

- [ ] **Step 1: Write failing tests in `tests/test_export.py`**

```python
import numpy as np
import xarray as xr

from server.export import to_csv_bytes, to_dataframe, to_netcdf_bytes
from server.subset import apply_filters


def test_to_dataframe_columns_and_row_count(open_sample):
    ds, coords = open_sample
    df = to_dataframe(ds, coords)
    assert list(df.columns) == ["time", "latitude", "longitude", "temperature", "humidity"]
    assert len(df) == 4 * 18 * 36


def test_to_dataframe_drops_fully_masked_rows(open_sample):
    ds, coords = open_sample
    filtered = apply_filters(
        ds, coords, var_filter={"variable": "temperature", "min": 20.0, "max": None}
    )
    df = to_dataframe(filtered, coords)
    assert (df["temperature"] >= 20.0).all()
    assert 0 < len(df) < 4 * 18 * 36


def test_to_csv_bytes_has_header(open_sample):
    ds, coords = open_sample
    data = to_csv_bytes(ds, coords)
    first_line = data.decode("utf-8").splitlines()[0]
    assert first_line == "time,latitude,longitude,temperature,humidity"


def test_netcdf_bytes_roundtrip(open_sample, tmp_path):
    ds, coords = open_sample
    subset = apply_filters(
        ds, coords, bbox={"west": -20.0, "south": -10.0, "east": 30.0, "north": 40.0}
    )
    data = to_netcdf_bytes(subset)
    out = tmp_path / "roundtrip.nc"
    out.write_bytes(data)
    with xr.open_dataset(out) as reopened:
        assert reopened.sizes == {"time": 4, "lat": 5, "lon": 5}
        assert set(reopened.data_vars) == {"temperature", "humidity"}
        np.testing.assert_allclose(
            reopened["temperature"].values, subset["temperature"].values
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_export.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'server.export'`.

- [ ] **Step 3: Write `server/export.py`**

```python
"""Turn a (filtered) dataset into downloadable netCDF or CSV bytes."""

import os
import tempfile


def to_netcdf_bytes(ds):
    """Write ds to netCDF and return the file contents.

    Goes through a temp file because the netCDF4 engine cannot write to an
    in-memory buffer; the in-memory path would silently fall back to the
    netCDF3 format with its size/type restrictions.
    """
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ds.to_netcdf(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def to_dataframe(ds, coords):
    """Long-format DataFrame: time, latitude, longitude, then one column per
    variable. Rows where every variable is NaN (masked out) are dropped."""
    df = ds.to_dataframe().reset_index()
    df = df.rename(columns={coords["lat"]: "latitude", coords["lon"]: "longitude"})
    data_cols = [str(name) for name in ds.data_vars]
    df = df.dropna(subset=data_cols, how="all")
    time = coords.get("time")
    lead = [c for c in (time, "latitude", "longitude") if c in df.columns]
    return df[lead + data_cols]


def to_csv_bytes(ds, coords):
    return to_dataframe(ds, coords).to_csv(index=False).encode("utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_export.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add server/export.py tests/test_export.py
git commit -m "feat: export subsets as netCDF bytes or time/lat/lon CSV"
```

---

### Task 11: FastAPI endpoints

**Files:**
- Create: `server/app.py`
- Create: `tests/test_api.py`

Four endpoints: `POST /api/open`, `GET /api/metadata`, `GET /api/slice` (PNG with value range in headers), `POST /api/export` (file download). One module-level `DatasetManager` holds the open file. The static frontend mounts at `/` **after** the API routes so `/api/*` always wins.

- [ ] **Step 1: Write failing tests in `tests/test_api.py`**

```python
import io

import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from server.app import app, manager

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    manager.close()  # isolate tests: manager is a module-level singleton


@pytest.fixture
def opened(client, sample_nc):
    res = client.post("/api/open", json={"path": str(sample_nc)})
    assert res.status_code == 200
    return client


def test_open_returns_metadata(client, sample_nc):
    res = client.post("/api/open", json={"path": str(sample_nc)})
    assert res.status_code == 200
    body = res.json()
    assert [v["name"] for v in body["variables"]] == ["temperature", "humidity"]
    assert body["time"]["count"] == 4


def test_open_missing_file_returns_404(client):
    res = client.post("/api/open", json={"path": "/nope/missing.nc"})
    assert res.status_code == 404


def test_open_invalid_file_returns_400(client, tmp_path):
    bad = tmp_path / "bad.nc"
    bad.write_text("not netcdf")
    res = client.post("/api/open", json={"path": str(bad)})
    assert res.status_code == 400


def test_metadata_requires_open_file(client):
    assert client.get("/api/metadata").status_code == 409


def test_slice_returns_png_with_value_range(opened):
    res = opened.get("/api/slice", params={"variable": "temperature", "time_index": 0})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content[:8] == PNG_MAGIC
    assert float(res.headers["x-vmin"]) < float(res.headers["x-vmax"])


def test_slice_unknown_variable_returns_404(opened):
    res = opened.get("/api/slice", params={"variable": "nope", "time_index": 0})
    assert res.status_code == 404


def test_slice_bad_time_index_returns_400(opened):
    res = opened.get("/api/slice", params={"variable": "temperature", "time_index": 99})
    assert res.status_code == 400


def test_export_csv_with_filters(opened):
    res = opened.post("/api/export", json={
        "format": "csv",
        "bbox": {"west": -20.0, "south": -10.0, "east": 30.0, "north": 40.0},
        "time_range": {"start": "2020-01-01", "end": "2020-01-02"},
    })
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment" in res.headers["content-disposition"]
    df = pd.read_csv(io.StringIO(res.text))
    assert list(df.columns) == ["time", "latitude", "longitude", "temperature", "humidity"]
    assert len(df) == 2 * 5 * 5
    assert df["latitude"].between(-10, 40).all()


def test_export_netcdf(opened, tmp_path):
    res = opened.post("/api/export", json={
        "format": "netcdf",
        "time_range": {"start": "2020-01-01", "end": "2020-01-01"},
    })
    assert res.status_code == 200
    out = tmp_path / "subset.nc"
    out.write_bytes(res.content)
    with xr.open_dataset(out) as ds:
        assert ds.sizes["time"] == 1
        assert "temperature" in ds.data_vars


def test_export_unknown_format_returns_400(opened):
    res = opened.post("/api/export", json={"format": "xlsx"})
    assert res.status_code == 400


def test_export_requires_open_file(client):
    res = client.post("/api/export", json={"format": "csv"})
    assert res.status_code == 409


def test_index_page_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_api.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'server.app'`.

- [ ] **Step 3: Write `server/app.py`**

```python
"""HTTP API and static frontend for the netCDF segmenter."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .dataset import DatasetManager
from .export import to_csv_bytes, to_netcdf_bytes
from .rendering import render_slice_png
from .subset import apply_filters

app = FastAPI(title="netCDF Segmenter & Exporter")
manager = DatasetManager()


class OpenRequest(BaseModel):
    path: str


class ExportRequest(BaseModel):
    format: str  # "csv" | "netcdf"
    bbox: dict | None = None
    polygon: list[list[float]] | None = None
    time_range: dict | None = None
    var_filter: dict | None = None


def _require_open():
    if manager.ds is None:
        raise HTTPException(status_code=409, detail="No dataset is open. POST /api/open first.")


@app.post("/api/open")
def open_dataset(req: OpenRequest):
    try:
        return manager.open(req.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/metadata")
def metadata():
    _require_open()
    return manager.metadata()


@app.get("/api/slice")
def slice_png(variable: str, time_index: int = 0):
    _require_open()
    if variable not in manager.ds.data_vars:
        raise HTTPException(status_code=404, detail=f"Unknown variable: {variable}")
    tname = manager.coords.get("time")
    if tname is not None:
        count = int(manager.ds.sizes[tname])
        if not 0 <= time_index < count:
            raise HTTPException(
                status_code=400,
                detail=f"time_index {time_index} out of range [0, {count - 1}]",
            )
    try:
        png, vmin, vmax = render_slice_png(manager.ds, manager.coords, variable, time_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Vmin": str(vmin), "X-Vmax": str(vmax)},
    )


@app.post("/api/export")
def export(req: ExportRequest):
    _require_open()
    if req.format not in ("csv", "netcdf"):
        raise HTTPException(status_code=400, detail=f"Unknown format: {req.format}")
    try:
        subset = apply_filters(
            manager.ds, manager.coords,
            bbox=req.bbox, polygon=req.polygon,
            time_range=req.time_range, var_filter=req.var_filter,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    stem = manager.path.stem + "_subset"
    if req.format == "csv":
        return Response(
            content=to_csv_bytes(subset, manager.coords),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'},
        )
    return Response(
        content=to_netcdf_bytes(subset),
        media_type="application/netcdf",
        headers={"Content-Disposition": f'attachment; filename="{stem}.nc"'},
    )


# Mounted last so /api/* routes take precedence.
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True),
    name="static",
)
```

- [ ] **Step 4: Run the whole suite**

```bash
.venv/bin/python -m pytest -v
```

Expected: `40 passed` (2 fixtures + 10 dataset + 2 rendering + 10 subset + 4 export + 12 api)

- [ ] **Step 5: Commit**

```bash
git add server/app.py tests/test_api.py
git commit -m "feat: FastAPI endpoints for open, metadata, slice PNG, and export"
```

---

### Task 12: Entrypoint and demo data script

**Files:**
- Create: `server/__main__.py`
- Create: `scripts/make_demo_data.py`

- [ ] **Step 1: Write `server/__main__.py`**

```python
"""Run the app: python -m server [optional/path/to/file.nc]"""

import sys

import uvicorn

from .app import app, manager


def main():
    if len(sys.argv) > 1:
        meta = manager.open(sys.argv[1])
        print(f"Opened {meta['path']} ({meta['size_bytes'] / 1e6:.1f} MB)")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/make_demo_data.py`**

```python
"""Generate data/demo_global.nc — a ~10 MB global demo file for manual testing."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OUT = Path(__file__).resolve().parent.parent / "data" / "demo_global.nc"


def main():
    lats = np.arange(-89.0, 90.0, 2.0)                       # 90 cells
    lons = np.arange(-179.0, 180.0, 2.0)                     # 180 cells
    times = pd.date_range("2020-01-01", periods=73, freq="5D")  # one year

    lon2d, lat2d = np.meshgrid(lons, lats)
    day = times.dayofyear.values[:, None, None]
    season = np.cos(2 * np.pi * (day - 196) / 365.25)        # +1 in N-hemisphere summer

    temperature = (
        32.0 * np.cos(np.radians(lat2d))[None, :, :]
        - 8.0
        + 12.0 * season * (lat2d / 90.0)[None, :, :]
    )
    precipitation = np.clip(
        8.0 * np.cos(np.radians(lat2d * 3.0))[None, :, :] ** 2
        + 4.0 * np.sin(np.radians(lon2d))[None, :, :] * season,
        0.0, None,
    )

    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lat", "lon"), temperature.astype("float32"),
                {"units": "degC", "long_name": "Surface air temperature"},
            ),
            "precipitation": (
                ("time", "lat", "lon"), precipitation.astype("float32"),
                {"units": "mm/day", "long_name": "Precipitation rate"},
            ),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
    )
    OUT.parent.mkdir(exist_ok=True)
    ds.to_netcdf(OUT)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the demo file and smoke-test the server**

```bash
.venv/bin/python scripts/make_demo_data.py
```

Expected: `Wrote .../data/demo_global.nc (9.5 MB)` (size approximate).

```bash
.venv/bin/python -m server data/demo_global.nc &
sleep 2
curl -s http://127.0.0.1:8000/api/metadata | head -c 300
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" \
  "http://127.0.0.1:8000/api/slice?variable=temperature&time_index=0"
kill %1
```

Expected: JSON starting with `{"path":` listing temperature/precipitation, then `200 image/png`.

- [ ] **Step 4: Run the full test suite (regression check)**

```bash
.venv/bin/python -m pytest
```

Expected: `40 passed`

- [ ] **Step 5: Commit**

```bash
git add server/__main__.py scripts/make_demo_data.py
git commit -m "feat: add server entrypoint and demo data generator"
```

---

### Task 13: Frontend — page structure and styling

**Files:**
- Modify: `static/index.html` (replace the Task 1 placeholder entirely)
- Create: `static/style.css`

Layout: fixed sidebar (file, display controls, filters, export) + full-height map. Leaflet and leaflet-draw come from unpkg CDN — no build step.

- [ ] **Step 1: Replace `static/index.html` with the full UI**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>netCDF Segmenter & Exporter</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css">
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <aside id="sidebar">
    <h1>netCDF Segmenter</h1>

    <section>
      <h2>File</h2>
      <input id="file-path" type="text" placeholder="/path/to/data.nc">
      <button id="open-btn">Open</button>
      <p id="file-info" class="muted"></p>
    </section>

    <section>
      <h2>Display</h2>
      <label>Variable
        <select id="variable-select"></select>
      </label>
      <label>Time step — <span id="time-label"></span>
        <input id="time-slider" type="range" min="0" max="0" value="0">
      </label>
      <div id="legend">
        <div id="legend-gradient"></div>
        <div id="legend-labels">
          <span id="legend-min"></span><span id="legend-max"></span>
        </div>
      </div>
    </section>

    <section>
      <h2>Filters</h2>
      <p class="muted">Use the toolbar on the map to draw a rectangle or polygon.</p>
      <p id="shape-info" class="muted">No shape drawn (whole globe).</p>
      <button id="clear-shape-btn">Clear shape</button>
      <label>Start time <input id="time-start" type="text"></label>
      <label>End time <input id="time-end" type="text"></label>
      <label>Filter variable
        <select id="filter-variable"><option value="">(none)</option></select>
      </label>
      <div class="row">
        <label>Min <input id="filter-min" type="number" step="any"></label>
        <label>Max <input id="filter-max" type="number" step="any"></label>
      </div>
    </section>

    <section>
      <h2>Export</h2>
      <button id="export-csv-btn">Download CSV</button>
      <button id="export-nc-btn">Download netCDF</button>
      <p id="status" class="muted"></p>
    </section>
  </aside>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `static/style.css`**

```css
:root { font-family: system-ui, sans-serif; }
body { margin: 0; display: flex; height: 100vh; }
#sidebar {
  width: 320px; overflow-y: auto; padding: 16px; box-sizing: border-box;
  background: #f7f7f7; border-right: 1px solid #ddd;
}
#map { flex: 1; }
h1 { font-size: 18px; margin: 0 0 12px; }
h2 {
  font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;
  color: #666; margin: 20px 0 8px;
}
section { margin-bottom: 8px; }
label { display: block; margin: 8px 0 4px; font-size: 13px; }
input[type="text"], input[type="number"], select {
  width: 100%; box-sizing: border-box; padding: 6px;
}
input[type="range"] { width: 100%; }
button { margin-top: 8px; padding: 6px 12px; cursor: pointer; }
.row { display: flex; gap: 8px; }
.row label { flex: 1; }
.muted { color: #888; font-size: 12px; }
#legend-gradient {
  height: 12px; border-radius: 3px; margin-top: 8px;
  /* viridis color stops, matching the backend's default colormap */
  background: linear-gradient(to right,
    #440154, #414487, #2a788e, #22a884, #7ad151, #fde725);
}
#legend-labels {
  display: flex; justify-content: space-between;
  font-size: 11px; color: #555;
}
```

- [ ] **Step 3: Verify the page is served and the suite still passes**

```bash
.venv/bin/python -m pytest tests/test_api.py::test_index_page_served -v
.venv/bin/python -m pytest
```

Expected: `1 passed`, then `40 passed`.

- [ ] **Step 4: Commit**

```bash
git add static/index.html static/style.css
git commit -m "feat: frontend page structure and styling for map and controls"
```

---

### Task 14: Frontend — map, drawing, and export logic

**Files:**
- Create: `static/app.js`

All interaction logic. Key flows: open file → populate controls → fetch slice PNG → overlay on map; draw rectangle/polygon → remember as the shape filter (one at a time); export → POST filters, download the blob.

- [ ] **Step 1: Write `static/app.js`**

```javascript
let metadata = null;
let overlay = null;
let drawnShape = null; // {type: "bbox", value: {...}} | {type: "polygon", value: [[lon,lat],...]}

const map = L.map("map").setView([20, 0], 2);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  noWrap: true,
}).addTo(map);

const drawnItems = new L.FeatureGroup().addTo(map);
map.addControl(new L.Control.Draw({
  draw: {
    rectangle: true,
    polygon: true,
    polyline: false,
    circle: false,
    marker: false,
    circlemarker: false,
  },
  edit: { featureGroup: drawnItems, edit: false, remove: false },
}));

map.on(L.Draw.Event.CREATED, (e) => {
  drawnItems.clearLayers(); // one shape at a time
  drawnItems.addLayer(e.layer);
  if (e.layerType === "rectangle") {
    const b = e.layer.getBounds();
    drawnShape = { type: "bbox", value: {
      west: b.getWest(), south: b.getSouth(),
      east: b.getEast(), north: b.getNorth(),
    }};
    setText("shape-info", "Bounding box selected.");
  } else {
    // first ring of the GeoJSON polygon, already in [lon, lat] order
    const ring = e.layer.toGeoJSON().geometry.coordinates[0];
    drawnShape = { type: "polygon", value: ring };
    setText("shape-info", "Polygon selected.");
  }
});

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

async function openFile() {
  const path = document.getElementById("file-path").value.trim();
  if (!path) return;
  setText("status", "Opening…");
  const res = await fetch("/api/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    setText("status", `Error: ${err.detail}`);
    return;
  }
  metadata = await res.json();
  setText("status", "");
  applyMetadata();
  await refreshOverlay();
}

function applyMetadata() {
  const sizeMb = (metadata.size_bytes / 1048576).toFixed(1);
  setText("file-info", `${metadata.path} (${sizeMb} MB)`);

  const varSel = document.getElementById("variable-select");
  const filterSel = document.getElementById("filter-variable");
  varSel.innerHTML = "";
  filterSel.innerHTML = '<option value="">(none)</option>';
  for (const v of metadata.variables) {
    const label = v.units ? `${v.name} (${v.units})` : v.name;
    varSel.add(new Option(label, v.name));
    filterSel.add(new Option(label, v.name));
  }

  const slider = document.getElementById("time-slider");
  const count = metadata.time ? metadata.time.count : 1;
  slider.max = String(Math.max(0, count - 1));
  slider.value = "0";
  updateTimeLabel();

  if (metadata.time) {
    document.getElementById("time-start").value = metadata.time.start;
    document.getElementById("time-end").value = metadata.time.end;
  }

  const ext = metadata.extent;
  map.fitBounds([[ext.south, ext.west], [ext.north, ext.east]]);
}

function updateTimeLabel() {
  const idx = Number(document.getElementById("time-slider").value);
  if (metadata && metadata.time && metadata.time.values) {
    setText("time-label", metadata.time.values[idx]);
  } else if (metadata && metadata.time) {
    setText("time-label", `index ${idx} of ${metadata.time.count - 1}`);
  } else {
    setText("time-label", "(no time axis)");
  }
}

async function refreshOverlay() {
  if (!metadata) return;
  const variable = document.getElementById("variable-select").value;
  const idx = document.getElementById("time-slider").value;
  const res = await fetch(
    `/api/slice?variable=${encodeURIComponent(variable)}&time_index=${idx}`
  );
  if (!res.ok) {
    setText("status", "Failed to render slice.");
    return;
  }
  setText("legend-min", Number(res.headers.get("X-Vmin")).toPrecision(4));
  setText("legend-max", Number(res.headers.get("X-Vmax")).toPrecision(4));
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const ext = metadata.extent;
  const bounds = [[ext.south, ext.west], [ext.north, ext.east]];
  if (overlay) {
    overlay.setUrl(url);
    overlay.setBounds(L.latLngBounds(bounds));
  } else {
    overlay = L.imageOverlay(url, bounds, { opacity: 0.75 }).addTo(map);
  }
}

function buildFilters() {
  const filters = {};
  if (drawnShape && drawnShape.type === "bbox") filters.bbox = drawnShape.value;
  if (drawnShape && drawnShape.type === "polygon") filters.polygon = drawnShape.value;

  const start = document.getElementById("time-start").value.trim();
  const end = document.getElementById("time-end").value.trim();
  if (start || end) filters.time_range = { start: start || null, end: end || null };

  const fvar = document.getElementById("filter-variable").value;
  if (fvar) {
    const min = document.getElementById("filter-min").value;
    const max = document.getElementById("filter-max").value;
    filters.var_filter = {
      variable: fvar,
      min: min === "" ? null : Number(min),
      max: max === "" ? null : Number(max),
    };
  }
  return filters;
}

async function exportSubset(format) {
  if (!metadata) {
    setText("status", "Open a file first.");
    return;
  }
  setText("status", "Preparing export…");
  const res = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format, ...buildFilters() }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    setText("status", `Export failed: ${err.detail}`);
    return;
  }
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const cd = res.headers.get("Content-Disposition") || "";
  const match = cd.match(/filename="(.+)"/);
  a.download = match ? match[1] : (format === "csv" ? "subset.csv" : "subset.nc");
  a.click();
  URL.revokeObjectURL(a.href);
  setText("status", "Export downloaded.");
}

document.getElementById("open-btn").addEventListener("click", openFile);
document.getElementById("variable-select").addEventListener("change", refreshOverlay);
document.getElementById("time-slider").addEventListener("input", updateTimeLabel);
document.getElementById("time-slider").addEventListener("change", refreshOverlay);
document.getElementById("clear-shape-btn").addEventListener("click", () => {
  drawnItems.clearLayers();
  drawnShape = null;
  setText("shape-info", "No shape drawn (whole globe).");
});
document.getElementById("export-csv-btn").addEventListener("click", () => exportSubset("csv"));
document.getElementById("export-nc-btn").addEventListener("click", () => exportSubset("netcdf"));
```

- [ ] **Step 2: Smoke-test in a browser**

```bash
.venv/bin/python -m server data/demo_global.nc
```

Open http://127.0.0.1:8000 and check the browser console for errors. (The Open-file field can also be used: enter the absolute path to `data/demo_global.nc`.) Stop the server with Ctrl-C when done. A fuller manual pass happens in Task 15.

- [ ] **Step 3: Commit**

```bash
git add static/app.js
git commit -m "feat: map interaction, shape drawing, time slider, and export downloads"
```

---

### Task 15: README and end-to-end manual verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# netCDF Segmenter & Exporter

Open a gridded spatio-temporal netCDF file, view any variable/time slice on a
zoomable world map, filter by drawn bounding box or polygon, time range, and
variable value range, then export the subset as netCDF or CSV.

Large files are handled lazily: only metadata and the currently displayed
slice are read until you export.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/python -m server [optional/path/to/file.nc]
```

Then open http://127.0.0.1:8000. You can also open a file from the UI by
entering its absolute path.

No demo data handy? Generate some:

```bash
.venv/bin/python scripts/make_demo_data.py
.venv/bin/python -m server data/demo_global.nc
```

## Export formats

- **netCDF** — the filtered subset; cells masked by polygon or value filters
  are stored as NaN.
- **CSV** — long format: `time, latitude, longitude, <one column per variable>`.
  Fully masked rows are dropped.

## Tests

```bash
.venv/bin/python -m pytest
```

## Known limitations

- One file at a time; shapes must not cross the antimeridian.
- Variables with extra dimensions (e.g. vertical levels) are listed but
  cannot be displayed or must be pre-flattened.
```

- [ ] **Step 2: Run the full test suite one final time**

```bash
.venv/bin/python -m pytest
```

Expected: `40 passed`

- [ ] **Step 3: Manual verification checklist**

Start the app (`.venv/bin/python -m server data/demo_global.nc`, open http://127.0.0.1:8000) and verify each item. The `c30d329f5814:webapp-testing` skill (Playwright) may be used to automate this if preferred.

- [ ] Temperature overlay appears over the world map; zoom and pan work.
- [ ] Switching the variable to `precipitation` redraws the overlay and updates the legend min/max.
- [ ] Moving the time slider updates the timestamp label; releasing it redraws the overlay (the demo file's temperature visibly shifts between January and July).
- [ ] Drawing a rectangle, then exporting CSV downloads a file whose latitude/longitude values fall inside the rectangle (spot-check in a text editor).
- [ ] Drawing a polygon, then exporting netCDF downloads a `.nc` cropped to the polygon's bounding box with NaN outside the shape (spot-check: `.venv/bin/python -c "import xarray; print(xarray.open_dataset('<download>'))"`).
- [ ] Setting a time range of a single month and exporting CSV yields only timestamps in that month.
- [ ] Setting filter variable `temperature` with Min `20` and exporting CSV yields only temperature values ≥ 20.
- [ ] "Clear shape" removes the drawn shape and subsequent exports cover the whole globe.
- [ ] Opening a nonexistent path shows an error message in the status line instead of breaking the page.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, usage, and limitations"
```

---

## Spec coverage map

| Spec requirement | Where |
|---|---|
| Large files: load only a portion + metadata (variables, full time range) | Task 5 (lazy `open_dataset`, metadata), Task 6 (per-slice reads) |
| One file at a time | Task 5 (`DatasetManager.open` replaces previous) |
| Single time slice on a zoom/pan global map | Tasks 6, 11, 13, 14 (PNG slice + Leaflet overlay) |
| Bounding box selection | Task 7 (backend), Task 14 (leaflet-draw rectangle) |
| Custom shape selection | Task 8 (backend), Task 14 (leaflet-draw polygon) |
| Time range filter | Task 7 (backend), Task 14 (UI inputs) |
| Filter by a variable (value range) | Task 9 (backend), Task 14 (UI inputs) |
| Export smaller netCDF | Task 10 (backend), Task 11 (endpoint), Task 14 (download) |
| Export CSV with latitude, longitude, and per-variable columns | Task 10 (column contract tested), Task 11, Task 14 |
| Choose variable to visualize | Task 5 (metadata variables), Tasks 11, 14 (selector) |
