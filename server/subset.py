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
