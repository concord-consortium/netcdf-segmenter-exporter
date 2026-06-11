"""Open netCDF files lazily and understand their coordinate layout."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LAT_NAMES = {"lat", "latitude"}
LON_NAMES = {"lon", "longitude"}
TIME_NAMES = {"time", "t"}
# CF-convention unit spellings, lowercased
LAT_UNITS = {"degrees_north", "degree_north", "degrees_n", "degree_n", "degreen", "degreesn"}
LON_UNITS = {"degrees_east", "degree_east", "degrees_e", "degree_e", "degreee", "degreese"}

# time steps per block when scanning a variable's global value range:
# bounds transient memory to roughly 2-3x the block's native size
# (~0.5 GB for a 596x1385 float32 grid at 64 steps: block + finite copy)
RANGE_SCAN_CHUNK = 64


def _match_coord(ds, names, std_names, units_set):
    """Find a 1D coordinate by name, then standard_name, then units.

    Separate passes so a name match anywhere in the dataset beats a
    units match on an earlier coordinate. Restricting to 1D coordinates
    keeps curvilinear/projected auxiliary coords (e.g. 2D nav_lat) out.
    """
    candidates = [(str(n), c) for n, c in ds.coords.items() if c.ndim == 1]
    for name, _ in candidates:
        if name.lower() in names:
            return name
    for name, coord in candidates:
        if str(coord.attrs.get("standard_name", "")).lower() in std_names:
            return name
    for name, coord in candidates:
        if str(coord.attrs.get("units", "")).lower() in units_set:
            return name
    return None


def detect_coords(ds):
    """Return {"lat": name, "lon": name, "time": name-or-None} for a dataset.

    Raises ValueError if latitude or longitude cannot be identified.
    Projected coordinates (e.g. x/y in meters) are deliberately rejected:
    this app does not reproject, so guessing would put meter values on a
    degree-based map.
    """
    lat = _match_coord(ds, LAT_NAMES, {"latitude"}, LAT_UNITS)
    lon = _match_coord(ds, LON_NAMES, {"longitude"}, LON_UNITS)
    time = None
    for name, coord in ds.coords.items():
        if (
            str(name).lower() in TIME_NAMES
            or str(coord.attrs.get("standard_name", "")).lower() == "time"
            or np.issubdtype(coord.dtype, np.datetime64)
        ):
            time = str(name)
            break
    if lat is None or lon is None:
        raise ValueError(
            "Could not identify latitude/longitude coordinates in this file"
        )
    return {"lat": lat, "lon": lon, "time": time}


def normalize_coords(ds, coords):
    """Return a view of ds with strictly ascending latitude and longitude
    in [-180, 180].

    xarray's sortby/assign_coords operate on the (small) coordinate arrays and
    keep data variables lazy, so this is cheap even for huge files.
    """
    lat, lon = coords["lat"], coords["lon"]
    if float(ds[lat][0]) > float(ds[lat][-1]):
        ds = ds.sortby(lat)
    lon_da = ds[lon]
    lon_vals = lon_da.values
    if float(lon_vals.max()) > 180.0:
        wrapped = ((lon_vals + 180.0) % 360.0) - 180.0
        # copy(data=...) keeps the original attrs (units: degrees_east)
        ds = ds.assign_coords({lon: lon_da.copy(data=wrapped)})
        ds = ds.sortby(lon)
        # a cyclic wrap column (a grid carrying both lon=0 and lon=360)
        # becomes a duplicate after wrapping; keep the first occurrence
        new_vals = ds[lon].values
        keep = np.ones(new_vals.size, dtype=bool)
        keep[1:] = new_vals[1:] != new_vals[:-1]
        if not keep.all():
            ds = ds.isel({lon: np.flatnonzero(keep)})
    if float(ds[lon][0]) > float(ds[lon][-1]):
        ds = ds.sortby(lon)
    return ds


def _iso(value):
    """Render a time coordinate value as an ISO-8601 string.

    Numeric values (undecoded CF time) pass through as plain strings
    rather than being misread as nanoseconds-since-epoch.
    """
    if isinstance(value, (np.datetime64, datetime)):
        return pd.Timestamp(value).isoformat()
    iso = getattr(value, "isoformat", None)  # cftime calendars
    if callable(iso):
        return iso()
    return str(value)


def _cell_edges(vals, vmin_limit, vmax_limit):
    """Extend cell-center coordinates outward by half a cell, clamped.

    Image overlays need cell EDGES; using centers shrinks the image by half
    a cell per side. Single-point axes get no padding (spacing unknown)."""
    lo, hi = float(vals[0]), float(vals[-1])
    if vals.size > 1:
        lo -= abs(float(vals[1]) - float(vals[0])) / 2.0
        hi += abs(float(vals[-1]) - float(vals[-2])) / 2.0
    return max(lo, vmin_limit), min(hi, vmax_limit)


class DatasetManager:
    """Holds the single currently-open dataset, lazily loaded."""

    def __init__(self):
        self.ds = None
        self.coords = None
        self.path = None
        self._ranges = {}

    def open(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            ds = xr.open_dataset(path)  # lazy: no data variables are read yet
        except PermissionError:
            raise  # surfaced distinctly: not a parse failure
        except Exception as exc:
            raise ValueError(f"Could not open {path} as netCDF: {exc}") from exc
        try:
            coords = detect_coords(ds)
            ds = normalize_coords(ds, coords)
        except Exception:
            ds.close()
            raise
        self.close()
        self.ds, self.coords, self.path = ds, coords, path
        return self.metadata()

    def close(self):
        if self.ds is not None:
            self.ds.close()
        self.ds = self.coords = self.path = None
        self._ranges = {}

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

        lat_vals = self.ds[lat].values
        lon_vals = self.ds[lon].values
        south, north = _cell_edges(lat_vals, -90.0, 90.0)
        west, east = _cell_edges(lon_vals, -180.0, 180.0)

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
            "edges": {"south": south, "north": north, "west": west, "east": east},
        }

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
            values = np.asarray(block.values)  # native dtype: no float64 copy
            finite = values[np.isfinite(values)]
            if finite.size:
                vmin = min(vmin, float(finite.min()))
                vmax = max(vmax, float(finite.max()))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 0.0, 1.0
        return vmin, vmax
