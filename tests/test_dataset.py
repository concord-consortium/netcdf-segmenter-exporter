import numpy as np
import pytest
import xarray as xr

from server.dataset import detect_coords


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
