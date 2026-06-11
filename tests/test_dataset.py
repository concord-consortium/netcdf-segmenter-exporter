import numpy as np
import pandas as pd
import pytest
import xarray as xr

from server.dataset import DatasetManager, detect_coords, normalize_coords


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


def test_manager_open_missing_file_raises():
    m = DatasetManager()
    with pytest.raises(FileNotFoundError):
        m.open("/nope/missing.nc")


def test_manager_open_non_netcdf_raises_value_error(tmp_path):
    bad = tmp_path / "bad.nc"
    bad.write_text("this is not a netCDF file")
    m = DatasetManager()
    with pytest.raises(ValueError):
        m.open(bad)


def test_manager_metadata(sample_nc):
    m = DatasetManager()
    meta = m.open(sample_nc)
    names = [v["name"] for v in meta["variables"]]
    assert names == ["temperature", "humidity"]
    assert meta["variables"][0]["units"] == "degC"
    assert meta["time"]["start"].startswith("2020-01-01")
    assert meta["time"]["end"].startswith("2020-01-04")
    assert meta["time"]["count"] == 4
    assert len(meta["time"]["values"]) == 4
    assert meta["extent"] == {
        "south": -85.0, "north": 85.0, "west": -175.0, "east": 175.0,
    }
    assert meta["size_bytes"] > 0
    m.close()


def test_manager_open_replaces_previous_file(sample_nc, rotated_nc):
    m = DatasetManager()
    m.open(sample_nc)
    meta = m.open(rotated_nc)
    assert meta["time"]["count"] == 2
    m.close()


def test_iso_numeric_time_values_pass_through():
    # numeric time (undecodable CF units) must not become 1970-epoch garbage
    from server.dataset import _iso
    assert _iso(731.0) == "731.0"
    assert _iso(0) == "0"


def test_manager_failed_open_preserves_previous(sample_nc, tmp_path):
    bad = tmp_path / "bad.nc"
    bad.write_text("not netcdf")
    m = DatasetManager()
    m.open(sample_nc)
    with pytest.raises(ValueError):
        m.open(bad)
    meta = m.metadata()  # previous dataset still open and queryable
    assert meta["path"] == str(sample_nc)
    assert meta["time"]["count"] == 4
    m.close()


def test_metadata_omits_time_values_when_huge(tmp_path):
    import pandas as pd
    import xarray as xr
    times = pd.date_range("2020-01-01", periods=2001, freq="h")
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.zeros((2001, 2, 2)))},
        coords={"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    path = tmp_path / "long.nc"
    ds.to_netcdf(path)
    ds.close()
    m = DatasetManager()
    meta = m.open(path)
    assert meta["time"]["count"] == 2001
    assert meta["time"]["values"] is None
    m.close()


def test_metadata_includes_cell_edge_bounds(sample_nc):
    # extent is cell centers; edges pads by half a cell for image overlay
    m = DatasetManager()
    meta = m.open(sample_nc)
    assert meta["edges"] == {
        "south": -90.0, "north": 90.0, "west": -180.0, "east": 180.0,
    }
    m.close()


def test_metadata_edges_clamped_to_globe(tmp_path):
    # near-pole centers must not produce edges beyond +/-90 / +/-180
    import pandas as pd
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.zeros((1, 3, 3)))},
        coords={
            "time": pd.date_range("2020-01-01", periods=1),
            "lat": [-89.0, 0.0, 89.0],
            "lon": [-179.0, 0.0, 179.0],
        },
    )
    path = tmp_path / "poles.nc"
    ds.to_netcdf(path)
    ds.close()
    m = DatasetManager()
    meta = m.open(path)
    assert meta["edges"]["south"] == -90.0 and meta["edges"]["north"] == 90.0
    assert meta["edges"]["west"] == -180.0 and meta["edges"]["east"] == 180.0
    m.close()


def test_manager_open_unreadable_file_raises_permission_error(sample_nc, tmp_path):
    import shutil

    locked = tmp_path / "locked.nc"
    shutil.copy(sample_nc, locked)
    locked.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            DatasetManager().open(locked)
    finally:
        locked.chmod(0o644)


def _write_range_file(tmp_path):
    """3 time steps; global min in step 0, global max in step 2, one inf."""
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    data = np.full((3, 2, 2), 10.0)
    data[0, 0, 0] = -5.0
    data[2, 1, 1] = 99.0
    data[1, 0, 1] = np.inf  # must be excluded from the range
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    path = tmp_path / "range.nc"
    ds.to_netcdf(path)
    ds.close()
    return path


def test_value_range_spans_all_time_steps_excluding_nonfinite(tmp_path):
    m = DatasetManager()
    m.open(_write_range_file(tmp_path))
    assert m.value_range("v") == (-5.0, 99.0)
    m.close()


def test_value_range_chunked_scan_matches(tmp_path, monkeypatch):
    import server.dataset as dataset_module

    # force one time step per block: min and max live in DIFFERENT blocks,
    # proving the cross-block reduction
    monkeypatch.setattr(dataset_module, "RANGE_SCAN_CHUNK", 1)
    m = DatasetManager()
    m.open(_write_range_file(tmp_path))
    assert m.value_range("v") == (-5.0, 99.0)
    m.close()


def test_value_range_caches_scan(sample_nc, monkeypatch):
    m = DatasetManager()
    m.open(sample_nc)
    calls = {"n": 0}
    real = m._scan_range

    def counting(variable):
        calls["n"] += 1
        return real(variable)

    monkeypatch.setattr(m, "_scan_range", counting)
    first = m.value_range("temperature")
    second = m.value_range("temperature")
    assert first == second
    assert calls["n"] == 1
    m.close()


def test_value_range_resets_on_new_open(sample_nc, rotated_nc):
    m = DatasetManager()
    m.open(sample_nc)
    m.value_range("temperature")
    m.open(rotated_nc)
    assert m._ranges == {}          # cache cleared with the old file
    lo, hi = m.value_range("temperature")
    assert lo < hi                  # recomputed for the new file
    m.close()


def test_value_range_all_nan_falls_back(tmp_path):
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.full((2, 2), np.nan))},
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    path = tmp_path / "allnan.nc"
    ds.to_netcdf(path)
    ds.close()
    m = DatasetManager()
    m.open(path)
    assert m.value_range("v") == (0.0, 1.0)
    m.close()


def test_value_range_unknown_variable_raises(sample_nc):
    m = DatasetManager()
    m.open(sample_nc)
    with pytest.raises(KeyError):
        m.value_range("nope")
    m.close()
