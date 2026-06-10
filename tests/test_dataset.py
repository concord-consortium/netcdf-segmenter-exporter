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
