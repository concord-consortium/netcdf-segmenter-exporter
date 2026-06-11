"""Render one time slice of one variable as a colormapped PNG."""

import io

import numpy as np
from matplotlib import image as mpimage


def render_slice_png(ds, coords, variable, time_index=0, cmap="viridis"):
    """Return (png_bytes, vmin, vmax) for variable at time_index.

    Only the requested slice is read from disk (xarray lazy indexing).
    NaN cells render transparent. Raises ValueError if the slice is not 2D
    (e.g. the variable has a vertical-level dimension).
    ds must have ascending latitude (as produced by normalize_coords); descending input would render vertically flipped.
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

    buf = io.BytesIO()
    # origin="lower": row 0 is the southernmost latitude (coords are ascending)
    mpimage.imsave(buf, data, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", format="png")
    return buf.getvalue(), vmin, vmax
