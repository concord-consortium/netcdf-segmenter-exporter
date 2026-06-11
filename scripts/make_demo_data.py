"""Generate data/demo_global.nc — a ~10 MB global demo file for manual testing."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OUT = Path(__file__).resolve().parent.parent / "data" / "demo_global.nc"


def main():
    lats = np.arange(-89.0, 90.0, 2.0)                          # 90 cells
    lons = np.arange(-179.0, 180.0, 2.0)                        # 180 cells
    times = pd.date_range("2020-01-01", periods=73, freq="5D")  # one year

    lon2d, lat2d = np.meshgrid(lons, lats)
    day = times.dayofyear.values[:, None, None]
    season = np.cos(2 * np.pi * (day - 196) / 365.25)           # +1 in NH summer

    temperature = (
        32.0 * np.cos(np.radians(lat2d))[None, :, :]
        - 8.0
        + 12.0 * season * (lat2d / 90.0)[None, :, :]
    )
    precipitation = np.clip(
        8.0 * np.cos(np.radians(lat2d * 3.0))[None, :, :] ** 2
        + 4.0 * np.sin(np.radians(lon2d))[None, :, :] * season,
        0.0, None,
    )

    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "lat", "lon"), temperature.astype("float32"),
                {"units": "degC", "long_name": "Surface air temperature"},
            ),
            "precipitation": (
                ("time", "lat", "lon"), precipitation.astype("float32"),
                {"units": "mm/day", "long_name": "Precipitation rate"},
            ),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
    )
    OUT.parent.mkdir(exist_ok=True)
    ds.to_netcdf(OUT)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
