"""Render one time slice of one variable as a colormapped PNG."""

import io
import math

import numpy as np
from matplotlib import image as mpimage

from .dataset import _cell_edges

# Web-Mercator display limit; Leaflet clamps overlay corners here too,
# so image rows and overlay bounds stay consistent for polar grids.
MAX_MERCATOR_LAT = 85.05112878
_MAX_OUTPUT_ROWS = 4096


def _mercator_y(lat_deg):
    rad = np.radians(lat_deg)
    return np.log(np.tan(np.pi / 4.0 + rad / 2.0))


def _resample_rows_for_mercator(data, lat_vals):
    """Re-space image rows for correct placement on a Web-Mercator map.

    Leaflet stretches the PNG linearly in screen space between its bounds,
    but screen y is mercator(lat), not latitude. Rows equally spaced in
    latitude sag toward the equator (over 1 degree across a CONUS-sized
    span). Resample with nearest-neighbor (keeps real data values) onto
    rows equally spaced in mercator y between the grid's cell edges.
    """
    south, north = _cell_edges(lat_vals, -90.0, 90.0)
    south = max(south, -MAX_MERCATOR_LAT)
    north = min(north, MAX_MERCATOR_LAT)
    if data.shape[0] < 2 or north <= south:
        return data
    # oversample so rows near the equator-side (compressed in mercator)
    # still map to at least one output row each
    widest = min(max(abs(south), abs(north)), MAX_MERCATOR_LAT)
    stretch = 1.0 / math.cos(math.radians(widest))
    n_out = int(math.ceil(data.shape[0] * stretch * 1.5))
    n_out = min(_MAX_OUTPUT_ROWS, max(data.shape[0], n_out))
    y_south, y_north = _mercator_y(south), _mercator_y(north)
    y_centers = y_south + (np.arange(n_out) + 0.5) * (y_north - y_south) / n_out
    target_lats = np.degrees(2.0 * np.arctan(np.exp(y_centers)) - np.pi / 2.0)
    # nearest source row for each output row (handles non-uniform grids)
    inner_edges = (lat_vals[:-1] + lat_vals[1:]) / 2.0
    idx = np.searchsorted(inner_edges, target_lats)
    return data[idx, :]


def render_slice_png(ds, coords, variable, time_index=0, cmap="viridis"):
    """Return (png_bytes, vmin, vmax) for variable at time_index.

    Only the requested slice is read from disk (xarray lazy indexing).
    NaN cells render transparent. Raises ValueError if the slice is not 2D
    (e.g. the variable has a vertical-level dimension).
    ds must have ascending latitude (as produced by normalize_coords); descending input would render vertically flipped.
    Image rows are resampled to be equally spaced in Web-Mercator y so that
    Leaflet's linear imageOverlay stretch places every row at its true latitude.
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
    data = np.array(da.values, dtype=float)
    # treat inf like missing data: transparent, and excluded from the scale
    data[~np.isfinite(data)] = np.nan

    finite = data[~np.isnan(data)]
    if finite.size:
        vmin = float(finite.min())
        vmax = float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1.0  # avoid a degenerate color scale

    lat_vals = np.asarray(ds[coords["lat"]].values, dtype=float)
    data = _resample_rows_for_mercator(data, lat_vals)

    buf = io.BytesIO()
    # origin="lower": row 0 is the southernmost latitude (coords are ascending)
    mpimage.imsave(buf, data, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", format="png")
    return buf.getvalue(), vmin, vmax
