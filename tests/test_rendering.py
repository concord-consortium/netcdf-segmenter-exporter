import math

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
    assert rgba[-1, 1, 3] == 0.0  # southernmost row (lat=0): masked cell transparent
    assert rgba[0, 1, 3] == 1.0   # northernmost row (lat=1): finite cell opaque
    assert rgba[0, 0, 3] == 1.0


def test_render_rows_positioned_for_mercator_display():
    """Leaflet stretches the PNG linearly in mercator screen space, so a band
    at a known latitude must land at the mercator-correct pixel row."""
    import io

    from matplotlib import image as mpimage

    lats = np.arange(24.5, 49.6, 0.5)    # CONUS-like span, 51 rows
    lons = np.arange(-120.0, -69.0, 1.0)
    band_lat = 44.0
    data = np.zeros((lats.size, lons.size))
    data[int(np.argmin(np.abs(lats - band_lat))), :] = 1.0
    ds = xr.Dataset({"v": (("lat", "lon"), data)}, coords={"lat": lats, "lon": lons})
    coords = {"lat": "lat", "lon": "lon", "time": None}
    png, _, _ = render_slice_png(ds, coords, "v")
    rgba = mpimage.imread(io.BytesIO(png))

    # the band is viridis-yellow on a dark background
    brightness = rgba[:, :, 0] + rgba[:, :, 1] - rgba[:, :, 2]
    band_row = int(np.argmax(brightness.mean(axis=1)))

    def merc(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    # cell edges: centers 24.5..49.5 step 0.5 -> edges 24.25 / 49.75
    south_edge, north_edge = 24.25, 49.75
    frac = (band_row + 0.5) / rgba.shape[0]   # decoded row 0 is the NORTH edge
    y = merc(north_edge) + frac * (merc(south_edge) - merc(north_edge))
    displayed_lat = math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)
    assert abs(displayed_lat - band_lat) < 0.5   # within one cell


def test_render_survives_polar_edge_grids():
    # global grids whose cell edges reach +/-90 must not hit mercator infinity
    lats = np.arange(-89.0, 90.0, 2.0)
    lons = np.arange(-179.0, 180.0, 2.0)
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.zeros((lats.size, lons.size)))},
        coords={"lat": lats, "lon": lons},
    )
    coords = {"lat": "lat", "lon": "lon", "time": None}
    png, _, _ = render_slice_png(ds, coords, "v")
    assert png[:8] == PNG_MAGIC


def test_render_with_explicit_range_scales_and_clips():
    import io

    from matplotlib import image as mpimage

    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.array([[0.0, 5.0], [10.0, 20.0]]))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    coords = {"lat": "lat", "lon": "lon", "time": None}
    png, vmin, vmax = render_slice_png(ds, coords, "v", vmin=0.0, vmax=10.0)
    assert (vmin, vmax) == (0.0, 10.0)  # echoes the explicit scale
    rgba = mpimage.imread(io.BytesIO(png))
    top_left = rgba[0, 0, :3]       # lat=1 row: value 10.0 == vmax
    top_right = rgba[0, 1, :3]      # value 20.0 must CLIP to the same color
    bottom_right = rgba[-1, 1, :3]  # lat=0 row: value 5.0 -> mid-scale
    np.testing.assert_allclose(top_left, top_right, atol=0.01)
    assert top_left[0] > 0.9 and top_left[1] > 0.85   # viridis top = yellow
    assert abs(float(bottom_right[0]) - float(top_left[0])) > 0.2  # mid != top


def test_render_partial_explicit_range_falls_back_to_slice():
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    coords = {"lat": "lat", "lon": "lon", "time": None}
    _, vmin, vmax = render_slice_png(ds, coords, "v", vmin=0.0)  # vmax omitted
    assert (vmin, vmax) == (1.0, 4.0)  # half-specified ranges are ignored
