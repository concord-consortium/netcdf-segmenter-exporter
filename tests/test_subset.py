import numpy as np
import pytest

from server.subset import apply_filters


def test_time_range_filter(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords, time_range={"start": "2020-01-02", "end": "2020-01-03"}
    )
    assert out.sizes["time"] == 2


def test_time_range_open_ended(open_sample):
    ds, coords = open_sample
    out = apply_filters(ds, coords, time_range={"start": "2020-01-03", "end": None})
    assert out.sizes["time"] == 2  # Jan 3 and Jan 4


def test_bbox_filter(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords, bbox={"west": -20.0, "south": -10.0, "east": 30.0, "north": 40.0}
    )
    # lat cells at -5,5,15,25,35 and lon cells at -15,-5,5,15,25
    assert out.sizes["lat"] == 5
    assert out.sizes["lon"] == 5
    assert float(out["lat"].min()) >= -10.0
    assert float(out["lat"].max()) <= 40.0
    assert float(out["lon"].min()) >= -20.0
    assert float(out["lon"].max()) <= 30.0


def test_no_filters_returns_full_dataset(open_sample):
    ds, coords = open_sample
    out = apply_filters(ds, coords)
    assert out.sizes == ds.sizes


def test_polygon_filter_crops_and_masks(open_sample):
    ds, coords = open_sample
    # triangle: base from (-60,-30) to (60,-30), apex at (0,60); [lon, lat] order
    polygon = [[-60.0, -30.0], [60.0, -30.0], [0.0, 60.0], [-60.0, -30.0]]
    out = apply_filters(ds, coords, polygon=polygon)

    # cropped to the triangle's bounding box
    assert float(out["lat"].min()) >= -30.0
    assert float(out["lat"].max()) <= 60.0
    assert float(out["lon"].min()) >= -60.0
    assert float(out["lon"].max()) <= 60.0

    # (lon=5, lat=5) is inside the triangle; (lon=-55, lat=55) is in the
    # bounding box but outside the triangle, so it must be masked
    inside = out["temperature"].isel(time=0).sel(lat=5.0, lon=5.0)
    outside = out["temperature"].isel(time=0).sel(lat=55.0, lon=-55.0)
    assert not np.isnan(float(inside))
    assert np.isnan(float(outside))


def test_polygon_mask_applies_to_all_variables(open_sample):
    ds, coords = open_sample
    polygon = [[-60.0, -30.0], [60.0, -30.0], [0.0, 60.0], [-60.0, -30.0]]
    out = apply_filters(ds, coords, polygon=polygon)
    t_nan = np.isnan(out["temperature"].values)
    h_nan = np.isnan(out["humidity"].values)
    assert np.array_equal(t_nan, h_nan)


def test_var_filter_min_only(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords, var_filter={"variable": "temperature", "min": 20.0, "max": None}
    )
    temp = out["temperature"].values
    assert np.nanmin(temp) >= 20.0
    assert np.isnan(temp).any()  # fixture spans 15..25, so some cells were masked


def test_var_filter_masks_all_variables(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords, var_filter={"variable": "temperature", "min": 20.0, "max": 24.0}
    )
    assert np.array_equal(
        np.isnan(out["temperature"].values), np.isnan(out["humidity"].values)
    )


def test_var_filter_unknown_variable_raises(open_sample):
    ds, coords = open_sample
    with pytest.raises(ValueError):
        apply_filters(ds, coords, var_filter={"variable": "nope", "min": 0, "max": 1})


def test_filters_compose(open_sample):
    ds, coords = open_sample
    out = apply_filters(
        ds, coords,
        bbox={"west": -20.0, "south": -10.0, "east": 30.0, "north": 40.0},
        time_range={"start": "2020-01-01", "end": "2020-01-02"},
        var_filter={"variable": "temperature", "min": 20.0, "max": None},
    )
    assert out.sizes == {"time": 2, "lat": 5, "lon": 5}
    assert np.nanmin(out["temperature"].values) >= 20.0
