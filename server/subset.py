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


def _mask_dataset(out, cond):
    """Apply .where(cond) only to variables whose dims can hold the mask.

    Dataset.where would broadcast the mask onto non-conforming variables —
    inflating CF bounds variables like time_bnds(time, nv) by lat*lon and
    giving static (lat, lon) variables a time dimension. Those variables
    pass through untouched instead.
    """
    masked = {
        name: da.where(cond)
        for name, da in out.data_vars.items()
        if set(cond.dims) <= set(da.dims)
    }
    return out.assign(masked)


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
    return _mask_dataset(out, mask)


def _apply_var_filter(out, var_filter):
    name = var_filter["variable"]
    if name not in out.data_vars:
        raise ValueError(f"Unknown filter variable: {name}")
    if var_filter.get("min") is None and var_filter.get("max") is None:
        return out
    da = out[name]
    cond = xr.ones_like(da, dtype=bool)
    if var_filter.get("min") is not None:
        cond = cond & (da >= var_filter["min"])
    if var_filter.get("max") is not None:
        cond = cond & (da <= var_filter["max"])
    return _mask_dataset(out, cond)
