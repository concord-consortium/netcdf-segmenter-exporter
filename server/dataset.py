"""Open netCDF files lazily and understand their coordinate layout."""

import numpy as np
import xarray as xr

LAT_NAMES = {"lat", "latitude"}
LON_NAMES = {"lon", "longitude"}
TIME_NAMES = {"time", "t"}
# CF-convention unit spellings, lowercased
LAT_UNITS = {"degrees_north", "degree_north", "degrees_n", "degree_n", "degreen", "degreesn"}
LON_UNITS = {"degrees_east", "degree_east", "degrees_e", "degree_e", "degreee", "degreese"}


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
