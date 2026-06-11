import numpy as np

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
