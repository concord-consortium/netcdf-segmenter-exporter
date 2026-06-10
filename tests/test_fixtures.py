import xarray as xr


def test_sample_fixture_shape(sample_nc):
    with xr.open_dataset(sample_nc) as ds:
        assert ds.sizes == {"time": 4, "lat": 18, "lon": 36}
        assert set(ds.data_vars) == {"temperature", "humidity"}


def test_rotated_fixture_conventions(rotated_nc):
    with xr.open_dataset(rotated_nc) as ds:
        assert float(ds["lat"][0]) > float(ds["lat"][-1])   # descending
        assert float(ds["lon"].max()) > 180                 # 0..360
