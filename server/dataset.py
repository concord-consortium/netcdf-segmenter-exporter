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
