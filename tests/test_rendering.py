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


def test_render_handles_inf_cells():
    import io

    from matplotlib import image as mpimage

    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.array([[1.0, np.inf], [3.0, 4.0]]))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    coords = {"lat": "lat", "lon": "lon", "time": None}
    png, vmin, vmax = render_slice_png(ds, coords, "v")
    assert png[:8] == PNG_MAGIC
    assert (vmin, vmax) == (1.0, 4.0)  # range from finite cells only
    # the inf cell renders transparent like NaN, not painted as a value
    rgba = mpimage.imread(io.BytesIO(png))
    assert rgba[1, 1, 3] == 0.0   # data[0,1] lands in image row 1 (origin="lower")
    assert rgba[0, 0, 3] == 1.0   # finite cells stay opaque
