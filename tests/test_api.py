import io
import json
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from server.app import app, manager

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    manager.close()  # isolate tests: manager is a module-level singleton


@pytest.fixture
def opened(client, sample_nc):
    res = client.post("/api/open", json={"path": str(sample_nc)})
    assert res.status_code == 200
    return client


def test_open_returns_metadata(client, sample_nc):
    res = client.post("/api/open", json={"path": str(sample_nc)})
    assert res.status_code == 200
    body = res.json()
    assert [v["name"] for v in body["variables"]] == ["temperature", "humidity"]
    assert body["time"]["count"] == 4


def test_open_missing_file_returns_404(client):
    res = client.post("/api/open", json={"path": "/nope/missing.nc"})
    assert res.status_code == 404


def test_open_invalid_file_returns_400(client, tmp_path):
    bad = tmp_path / "bad.nc"
    bad.write_text("not netcdf")
    res = client.post("/api/open", json={"path": str(bad)})
    assert res.status_code == 400


def test_metadata_requires_open_file(client):
    assert client.get("/api/metadata").status_code == 409


def test_slice_returns_png_with_value_range(opened):
    res = opened.get("/api/slice", params={"variable": "temperature", "time_index": 0})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content[:8] == PNG_MAGIC
    assert float(res.headers["x-vmin"]) < float(res.headers["x-vmax"])


def test_slice_etag_revalidation(opened):
    params = {"variable": "temperature", "time_index": 1}
    first = opened.get("/api/slice", params=params)
    etag = first.headers["etag"]
    second = opened.get("/api/slice", params=params, headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.content == b""


def test_slice_unknown_variable_returns_404(opened):
    res = opened.get("/api/slice", params={"variable": "nope", "time_index": 0})
    assert res.status_code == 404


def test_slice_bad_time_index_returns_400(opened):
    res = opened.get("/api/slice", params={"variable": "temperature", "time_index": 99})
    assert res.status_code == 400
    res = opened.get("/api/slice", params={"variable": "temperature", "time_index": -1})
    assert res.status_code == 400


def test_export_csv_with_filters(opened):
    res = opened.post("/api/export", json={
        "format": "csv",
        "bbox": {"west": -20.0, "south": -10.0, "east": 30.0, "north": 40.0},
        "time_range": {"start": "2020-01-01", "end": "2020-01-02"},
    })
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment" in res.headers["content-disposition"]
    df = pd.read_csv(io.StringIO(res.text))
    assert list(df.columns) == ["time", "latitude", "longitude", "temperature", "humidity"]
    assert len(df) == 2 * 5 * 5
    assert df["latitude"].between(-10, 40).all()


def test_export_netcdf(opened, tmp_path):
    res = opened.post("/api/export", json={
        "format": "netcdf",
        "time_range": {"start": "2020-01-01", "end": "2020-01-01"},
    })
    assert res.status_code == 200
    out = tmp_path / "subset.nc"
    out.write_bytes(res.content)
    with xr.open_dataset(out) as ds:
        assert ds.sizes["time"] == 1
        assert "temperature" in ds.data_vars


def test_export_unknown_format_returns_400(opened):
    res = opened.post("/api/export", json={"format": "xlsx"})
    assert res.status_code == 400


def test_export_unknown_filter_variable_returns_400(opened):
    res = opened.post("/api/export", json={
        "format": "csv",
        "var_filter": {"variable": "nope", "min": 0, "max": 1},
    })
    assert res.status_code == 400


def test_export_empty_selection_returns_400(opened):
    res = opened.post("/api/export", json={
        "format": "csv",
        "bbox": {"west": 100.0, "south": 86.0, "east": 110.0, "north": 89.0},
    })
    assert res.status_code == 400  # grid latitudes stop at 85


def test_export_csv_size_guard_returns_413(opened, monkeypatch):
    import server.app as app_module
    monkeypatch.setattr(app_module, "MAX_CSV_CELLS", 10)
    res = opened.post("/api/export", json={"format": "csv"})
    assert res.status_code == 413


def test_export_requires_open_file(client):
    res = client.post("/api/export", json={"format": "csv"})
    assert res.status_code == 409


def test_index_page_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_export_malformed_bbox_returns_422(opened):
    res = opened.post("/api/export", json={"format": "csv", "bbox": {"west": -10.0}})
    assert res.status_code == 422


def test_export_bad_time_string_returns_400(opened):
    res = opened.post(
        "/api/export",
        json={"format": "csv", "time_range": {"start": "garbage", "end": None}},
    )
    assert res.status_code == 400


def test_export_two_point_polygon_returns_422(opened):
    res = opened.post(
        "/api/export", json={"format": "csv", "polygon": [[0.0, 0.0], [1.0, 1.0]]}
    )
    assert res.status_code == 422


def test_slice_after_file_deleted_returns_409(client, tmp_path, sample_nc):
    import shutil

    moved = tmp_path / "moved.nc"
    shutil.copy(sample_nc, moved)
    assert client.post("/api/open", json={"path": str(moved)}).status_code == 200
    moved.unlink()
    res = client.get("/api/slice", params={"variable": "temperature", "time_index": 0})
    assert res.status_code == 409


def test_metadata_after_file_deleted_returns_409(client, tmp_path, sample_nc):
    import shutil

    moved = tmp_path / "moved2.nc"
    shutil.copy(sample_nc, moved)
    assert client.post("/api/open", json={"path": str(moved)}).status_code == 200
    moved.unlink()
    assert client.get("/api/metadata").status_code == 409


def test_open_unreadable_file_returns_403(client, sample_nc, tmp_path):
    import shutil

    locked = tmp_path / "locked.nc"
    shutil.copy(sample_nc, locked)
    locked.chmod(0o000)
    try:
        res = client.post("/api/open", json={"path": str(locked)})
        assert res.status_code == 403
        assert "permission" in res.json()["detail"].lower()
    finally:
        locked.chmod(0o644)


def test_slice_etag_changes_with_renderer_version(opened, monkeypatch):
    import server.app as app_module

    params = {"variable": "temperature", "time_index": 0}
    etag_before = opened.get("/api/slice", params=params).headers["etag"]
    monkeypatch.setattr(app_module, "RENDER_VERSION", "test-bump")
    etag_after = opened.get("/api/slice", params=params).headers["etag"]
    assert etag_after != etag_before
    # a stale etag from the old renderer must NOT revalidate to 304
    res = opened.get(
        "/api/slice", params=params, headers={"If-None-Match": etag_before}
    )
    assert res.status_code == 200


def test_browse_lists_directory(client, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.nc").write_bytes(b"xx")
    res = client.get("/api/browse", params={"path": str(tmp_path)})
    assert res.status_code == 200
    body = res.json()
    assert body["path"] == str(tmp_path.resolve())
    assert body["parent"] == str(tmp_path.resolve().parent)
    assert [d["name"] for d in body["dirs"]] == ["sub"]
    assert [f["name"] for f in body["files"]] == ["a.nc"]


def test_browse_defaults_to_home_and_needs_no_open_file(client):
    # note: no dataset is open in this fixture — browse must work anyway
    res = client.get("/api/browse")
    assert res.status_code == 200
    assert res.json()["path"] == str(Path.home())


def test_browse_missing_dir_returns_404(client):
    res = client.get("/api/browse", params={"path": "/nope/missing"})
    assert res.status_code == 404


def test_browse_file_path_returns_400(client, sample_nc):
    res = client.get("/api/browse", params={"path": str(sample_nc)})
    assert res.status_code == 400


def test_browse_unreadable_dir_returns_403(client, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        res = client.get("/api/browse", params={"path": str(locked)})
        assert res.status_code == 403
        assert "permission" in res.json()["detail"].lower()
    finally:
        locked.chmod(0o755)


def test_slice_value_range_fixed_across_time(opened):
    h0 = opened.get(
        "/api/slice", params={"variable": "temperature", "time_index": 0}
    ).headers
    h3 = opened.get(
        "/api/slice", params={"variable": "temperature", "time_index": 3}
    ).headers
    assert h0["x-vmin"] == h3["x-vmin"]
    assert h0["x-vmax"] == h3["x-vmax"]
    # global range over all 4 steps of uniform-[15,25) data
    lo, hi = float(h0["x-vmin"]), float(h0["x-vmax"])
    assert 15.0 <= lo < 16.0
    assert 24.0 < hi < 25.0


def test_value_range_stream_progress_then_result(opened):
    res = opened.get("/api/value-range", params={"variable": "temperature"})
    assert res.status_code == 200
    lines = [json.loads(l) for l in res.text.strip().splitlines()]
    assert len(lines) >= 2  # at least one progress line, then the final line
    final = lines[-1]
    assert "vmin" in final and "vmax" in final
    dones = [s["done"] for s in lines[:-1]]
    for step in lines[:-1]:
        assert 1 <= step["done"] <= step["total"]
    assert dones == sorted(dones)
    # second request: range is cached -> only the final line
    res2 = opened.get("/api/value-range", params={"variable": "temperature"})
    lines2 = [json.loads(l) for l in res2.text.strip().splitlines()]
    assert len(lines2) == 1
    assert lines2[0] == final


def test_value_range_stream_unknown_variable_404(opened):
    res = opened.get("/api/value-range", params={"variable": "nope"})
    assert res.status_code == 404


def test_value_range_stream_requires_open_file(client):
    res = client.get("/api/value-range", params={"variable": "temperature"})
    assert res.status_code == 409


def test_value_range_stream_no_store_header(opened):
    res = opened.get("/api/value-range", params={"variable": "humidity"})
    assert res.headers["cache-control"] == "no-store"
