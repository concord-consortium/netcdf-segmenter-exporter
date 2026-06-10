import numpy as np
import pytest
import xarray as xr

from server.dataset import detect_coords, normalize_coords


def test_detect_coords_standard_names(sample_nc):
    with xr.open_dataset(sample_nc) as ds:
        assert detect_coords(ds) == {"lat": "lat", "lon": "lon", "time": "time"}


def test_detect_coords_by_long_names():
    ds = xr.Dataset(
        {"v": (("latitude", "longitude"), np.zeros((2, 2)))},
        coords={
            "latitude": ("latitude", [0.0, 1.0], {"units": "degrees_north"}),
            "longitude": ("longitude", [0.0, 1.0], {"units": "degrees_east"}),
        },
    )
    assert detect_coords(ds) == {"lat": "latitude", "lon": "longitude", "time": None}


def test_detect_coords_by_units_only():
    # non-standard coord names: only the units attribute identifies them
    ds = xr.Dataset(
        {"v": (("grid_y", "grid_x"), np.zeros((2, 2)))},
        coords={
            "grid_y": ("grid_y", [0.0, 1.0], {"units": "degrees_north"}),
            "grid_x": ("grid_x", [0.0, 1.0], {"units": "degrees_east"}),
        },
    )
    assert detect_coords(ds) == {"lat": "grid_y", "lon": "grid_x", "time": None}


def test_detect_coords_raises_without_latlon():
    ds = xr.Dataset(
        {"v": (("a", "b"), np.zeros((2, 2)))},
        coords={"a": [0, 1], "b": [0, 1]},
    )
    with pytest.raises(ValueError):
        detect_coords(ds)


def test_detect_coords_by_standard_name():
    ds = xr.Dataset(
        {"v": (("yc", "xc"), np.zeros((2, 2)))},
        coords={
            "yc": ("yc", [0.0, 1.0], {"standard_name": "latitude"}),
            "xc": ("xc", [0.0, 1.0], {"standard_name": "longitude"}),
        },
    )
    assert detect_coords(ds) == {"lat": "yc", "lon": "xc", "time": None}


def test_detect_coords_rejects_projected_xy_in_meters():
    ds = xr.Dataset(
        {"v": (("y", "x"), np.zeros((2, 2)))},
        coords={
            "y": ("y", [0.0, 1000.0], {"units": "m"}),
            "x": ("x", [0.0, 1000.0], {"units": "m"}),
        },
    )
    with pytest.raises(ValueError):
        detect_coords(ds)


def test_detect_coords_name_match_beats_units_match():
    # decoy coord with degree units listed first must not shadow the real,
    # properly named coords
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.zeros((2, 2)))},
        coords={
            "decoy": ("lat", [9.0, 9.5], {"units": "degrees_north"}),
            "lat": ("lat", [0.0, 1.0]),
            "lon": ("lon", [0.0, 1.0]),
        },
    )
    coords = detect_coords(ds)
    assert coords["lat"] == "lat"
    assert coords["lon"] == "lon"


def test_detect_coords_time_by_dtype_with_nonstandard_name():
    ds = xr.Dataset(
        {"v": (("forecast_time", "lat", "lon"), np.zeros((2, 2, 2)))},
        coords={
            "forecast_time": np.array(
                ["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"
            ),
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
        },
    )
    assert detect_coords(ds)["time"] == "forecast_time"


def test_normalize_sorts_lat_ascending_and_wraps_lon(rotated_nc):
    with xr.open_dataset(rotated_nc) as ds:
        coords = detect_coords(ds)
        out = normalize_coords(ds, coords)
        lats = out["lat"].values
        lons = out["lon"].values
        assert (np.diff(lats) > 0).all()
        assert (np.diff(lons) > 0).all()
        assert lons.min() >= -180.0 and lons.max() <= 180.0


def test_normalize_preserves_values(rotated_nc):
    with xr.open_dataset(rotated_nc) as orig:
        coords = detect_coords(orig)
        out = normalize_coords(orig, coords)
        # lon 355 in the 0..360 file becomes lon -5 after normalization
        expected = float(orig["temperature"].isel(time=0).sel(lat=85.0, lon=355.0))
        actual = float(out["temperature"].isel(time=0).sel(lat=85.0, lon=-5.0))
        assert actual == expected


def test_normalize_leaves_wellbehaved_file_alone(sample_nc):
    with xr.open_dataset(sample_nc) as ds:
        coords = detect_coords(ds)
        out = normalize_coords(ds, coords)
        assert (out["lat"].values == ds["lat"].values).all()
        assert (out["lon"].values == ds["lon"].values).all()


def test_normalize_sorts_descending_lon_in_standard_convention():
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.arange(6.0).reshape(2, 3))},
        coords={"lat": [0.0, 10.0], "lon": [20.0, 10.0, 0.0]},
    )
    out = normalize_coords(ds, {"lat": "lat", "lon": "lon", "time": None})
    assert (np.diff(out["lon"].values) > 0).all()
    # the value follows its coordinate through the sort
    assert float(out["v"].sel(lat=0.0, lon=20.0)) == 0.0


def test_normalize_preserves_lon_attrs_when_wrapping(rotated_nc):
    with xr.open_dataset(rotated_nc) as ds:
        coords = detect_coords(ds)
        out = normalize_coords(ds, coords)
        assert out["lon"].attrs.get("units") == "degrees_east"


def test_normalize_drops_duplicate_cyclic_lon():
    lons = np.arange(0.0, 361.0, 30.0)  # 13 values: includes both 0 and 360
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.zeros((2, lons.size)))},
        coords={"lat": [0.0, 10.0], "lon": lons},
    )
    out = normalize_coords(ds, {"lat": "lat", "lon": "lon", "time": None})
    vals = out["lon"].values
    assert vals.size == lons.size - 1   # duplicate dropped
    assert (np.diff(vals) > 0).all()    # strictly ascending
