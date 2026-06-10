import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _build_dataset(lats, lons, times):
    rng = np.random.default_rng(42)
    shape = (len(times), len(lats), len(lons))
    temperature = 15 + 10 * rng.random(shape)   # uniform in [15, 25)
    humidity = 100 * rng.random(shape)          # uniform in [0, 100)
    return xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lat", "lon"),
                temperature,
                {"units": "degC", "long_name": "Air Temperature"},
            ),
            "humidity": (
                ("time", "lat", "lon"),
                humidity,
                {"units": "%", "long_name": "Relative Humidity"},
            ),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
    )


@pytest.fixture
def sample_nc(tmp_path):
    """Well-behaved file: ascending lat, -180..180 lon, 4 daily steps."""
    lats = np.arange(-85.0, 86.0, 10.0)      # 18 cells
    lons = np.arange(-175.0, 176.0, 10.0)    # 36 cells
    times = pd.date_range("2020-01-01", periods=4, freq="D")
    ds = _build_dataset(lats, lons, times)
    path = tmp_path / "sample.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


@pytest.fixture
def rotated_nc(tmp_path):
    """0..360 longitudes and descending latitudes, as in many model outputs."""
    lats = np.arange(85.0, -86.0, -10.0)     # 18 cells, descending
    lons = np.arange(5.0, 356.0, 10.0)       # 36 cells, 0..360 convention
    times = pd.date_range("2020-01-01", periods=2, freq="D")
    ds = _build_dataset(lats, lons, times)
    path = tmp_path / "rotated.nc"
    ds.to_netcdf(path)
    ds.close()
    return path
