"""Turn a (filtered) dataset into downloadable netCDF or CSV bytes."""

import os
import tempfile


def to_netcdf_bytes(ds):
    """Write ds to netCDF and return the file contents.

    Goes through a temp file because the netCDF4 engine cannot write to an
    in-memory buffer; the in-memory path would silently fall back to the
    netCDF3 format with its size/type restrictions.

    Note: variables masked by polygon/value filters are written as NaN;
    integer variables that passed through a mask were promoted to float.
    """
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ds.to_netcdf(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def to_dataframe(ds, coords):
    """Long-format DataFrame: time, latitude, longitude, then one column per
    variable. Rows where every variable is NaN (masked out) are dropped.

    Only variables living on the (time, lat, lon) grid are exported.
    Auxiliary variables with other dims (e.g. CF bounds like time_bnds)
    would cross-product their extra dims into duplicated rows.
    """
    lat, lon, time = coords["lat"], coords["lon"], coords.get("time")
    grid_dims = {d for d in (time, lat, lon) if d is not None}
    keep = [n for n, da in ds.data_vars.items() if set(da.dims) <= grid_dims]
    df = ds[keep].to_dataframe().reset_index()
    rename = {lat: "latitude", lon: "longitude"}
    if time is not None:
        rename[time] = "time"
    df = df.rename(columns=rename)
    data_cols = [str(n) for n in keep]
    df = df.dropna(subset=data_cols, how="all")
    lead = [c for c in ("time", "latitude", "longitude") if c in df.columns]
    return df[lead + data_cols]


def to_csv_bytes(ds, coords):
    return to_dataframe(ds, coords).to_csv(index=False).encode("utf-8")
