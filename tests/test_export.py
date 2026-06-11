import numpy as np
import xarray as xr

from server.export import to_csv_bytes, to_dataframe, to_netcdf_bytes
from server.subset import apply_filters


def test_to_dataframe_columns_and_row_count(open_sample):
    ds, coords = open_sample
    df = to_dataframe(ds, coords)
    assert list(df.columns) == ["time", "latitude", "longitude", "temperature", "humidity"]
    assert len(df) == 4 * 18 * 36


def test_to_dataframe_drops_fully_masked_rows(open_sample):
    ds, coords = open_sample
    filtered = apply_filters(
        ds, coords, var_filter={"variable": "temperature", "min": 20.0, "max": None}
    )
    df = to_dataframe(filtered, coords)
    assert (df["temperature"] >= 20.0).all()
    assert 0 < len(df) < 4 * 18 * 36


def test_to_csv_bytes_has_header(open_sample):
    ds, coords = open_sample
    data = to_csv_bytes(ds, coords)
    first_line = data.decode("utf-8").splitlines()[0]
    assert first_line == "time,latitude,longitude,temperature,humidity"


def test_netcdf_bytes_roundtrip(open_sample, tmp_path):
    ds, coords = open_sample
    subset = apply_filters(
        ds, coords, bbox={"west": -20.0, "south": -10.0, "east": 30.0, "north": 40.0}
    )
    data = to_netcdf_bytes(subset)
    out = tmp_path / "roundtrip.nc"
    out.write_bytes(data)
    with xr.open_dataset(out) as reopened:
        assert reopened.sizes == {"time": 4, "lat": 5, "lon": 5}
        assert set(reopened.data_vars) == {"temperature", "humidity"}
        np.testing.assert_allclose(
            reopened["temperature"].values, subset["temperature"].values
        )
