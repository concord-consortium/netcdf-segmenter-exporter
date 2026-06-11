"""HTTP API and static frontend for the netCDF segmenter."""

import hashlib
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dataset import DatasetManager
from .export import to_csv_bytes, to_netcdf_bytes
from .rendering import render_slice_png
from .subset import apply_filters

# CSV materializes one row per grid cell; past ~25M rows the in-memory
# DataFrame plus the CSV string risk exhausting laptop RAM.
MAX_CSV_CELLS = 25_000_000

app = FastAPI(title="netCDF Segmenter & Exporter")
manager = DatasetManager()


class OpenRequest(BaseModel):
    path: str


class BBox(BaseModel):
    west: float
    south: float
    east: float
    north: float


class TimeRange(BaseModel):
    start: str | None = None
    end: str | None = None


class VarFilter(BaseModel):
    variable: str
    min: float | None = None
    max: float | None = None


class ExportRequest(BaseModel):
    format: str  # "csv" | "netcdf"
    bbox: BBox | None = None
    polygon: list[tuple[float, float]] | None = Field(default=None, min_length=3)
    time_range: TimeRange | None = None
    var_filter: VarFilter | None = None


def _require_open():
    if manager.ds is None:
        raise HTTPException(
            status_code=409, detail="No dataset is open. POST /api/open first."
        )


# Handlers are deliberately async def: FastAPI then runs them on the event
# loop (not a threadpool), serializing requests so an open() can never close
# the dataset while another request is reading it. netCDF4 is not thread-safe.


@app.post("/api/open")
async def open_dataset(req: OpenRequest):
    try:
        return manager.open(req.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/metadata")
async def metadata():
    _require_open()
    return manager.metadata()


def _slice_etag(variable, time_index):
    try:
        st = manager.path.stat()
    except OSError:
        raise HTTPException(
            status_code=409,
            detail="The open file no longer exists on disk; open a file again.",
        )
    key = f"{manager.path}:{st.st_mtime_ns}:{st.st_size}:{variable}:{time_index}"
    return '"' + hashlib.sha1(key.encode()).hexdigest() + '"'


@app.get("/api/slice")
async def slice_png(request: Request, variable: str, time_index: int = 0):
    _require_open()
    if variable not in manager.ds.data_vars:
        raise HTTPException(status_code=404, detail=f"Unknown variable: {variable}")
    tname = manager.coords.get("time")
    if tname is not None:
        count = int(manager.ds.sizes[tname])
        if not 0 <= time_index < count:
            raise HTTPException(
                status_code=400,
                detail=f"time_index {time_index} out of range [0, {count - 1}]",
            )
    etag = _slice_etag(variable, time_index)
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    try:
        png, vmin, vmax = render_slice_png(
            manager.ds, manager.coords, variable, time_index
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    headers.update({"X-Vmin": str(vmin), "X-Vmax": str(vmax)})
    return Response(content=png, media_type="image/png", headers=headers)


@app.post("/api/export")
async def export(req: ExportRequest):
    _require_open()
    if req.format not in ("csv", "netcdf"):
        raise HTTPException(status_code=400, detail=f"Unknown format: {req.format}")
    try:
        subset = apply_filters(
            manager.ds, manager.coords,
            bbox=req.bbox.model_dump() if req.bbox else None,
            polygon=[list(p) for p in req.polygon] if req.polygon else None,
            time_range=req.time_range.model_dump() if req.time_range else None,
            var_filter=req.var_filter.model_dump() if req.var_filter else None,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid filter parameters: {exc}")

    cells = 1
    for dim in manager.coords.values():
        if dim is not None and dim in subset.sizes:
            cells *= int(subset.sizes[dim])
    if cells == 0:
        raise HTTPException(status_code=400, detail="Selection contains no grid cells")
    if req.format == "csv" and cells > MAX_CSV_CELLS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Subset would produce {cells:,} CSV rows "
                f"(limit {MAX_CSV_CELLS:,}); narrow the selection"
            ),
        )

    stem = manager.path.stem + "_subset"
    if req.format == "csv":
        return Response(
            content=to_csv_bytes(subset, manager.coords),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'},
        )
    return Response(
        content=to_netcdf_bytes(subset),
        media_type="application/netcdf",
        headers={"Content-Disposition": f'attachment; filename="{stem}.nc"'},
    )


# Mounted last so /api/* routes take precedence.
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True),
    name="static",
)
