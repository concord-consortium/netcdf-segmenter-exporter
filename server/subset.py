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


def _apply_var_filter(out, var_filter):
    raise NotImplementedError  # implemented two tasks from now (TDD)
