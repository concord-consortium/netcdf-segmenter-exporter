# netCDF Segmenter & Exporter

Open a gridded spatio-temporal netCDF file, view any variable/time slice on a
zoomable world map, filter by drawn bounding box or polygon, time range, and
variable value range, then export the subset as netCDF or CSV.

Large files are handled lazily: only metadata and the currently displayed
slice are read until you export.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/python -m server [optional/path/to/file.nc]
```

Then open http://127.0.0.1:8000. You can also open a file from the UI by
entering its absolute path.

No demo data handy? Generate some:

```bash
.venv/bin/python scripts/make_demo_data.py
.venv/bin/python -m server data/demo_global.nc
```

(Stop the server before regenerating demo data — an open netCDF handle on
macOS holds an HDF5 lock and the generator will fail with PermissionError.)

## Using the app

1. Enter a file path and click **Open** — variables, the time range, and the
   map extent populate from the file's metadata.
2. Pick a variable and scrub the time slider; the slice renders as a colored
   overlay (legend shows the slice's value range).
3. Filter spatially with the map's draw toolbar (rectangle or polygon — one
   shape at a time), temporally with the start/end inputs, and by value with
   the filter variable + min/max inputs.
4. **Download CSV** or **Download netCDF** exports the filtered subset.

## Export formats

- **netCDF** — the filtered subset; cells masked by polygon or value filters
  are stored as NaN. Variables that went through a mask are promoted to
  float (netCDF cannot store NaN in integer types). Auxiliary variables that
  don't share the grid dims (e.g. CF `time_bnds`) pass through unmasked.
- **CSV** — long format: `time, latitude, longitude, <one column per
  variable>`. Only variables on the (time, lat, lon) grid are included;
  rows where every variable is masked are dropped. CSV exports are capped at
  25 million rows — narrow the selection if you hit the limit.

## Tests

```bash
.venv/bin/python -m pytest
```

## Known limitations

- One file at a time; coordinates must be a regular 1D lat/lon grid
  (projected x/y grids and 2D curvilinear coordinates are rejected rather
  than guessed at).
- Shapes that cross the antimeridian aren't supported (the UI warns and
  clears the shape). Source grids that themselves span the antimeridian
  discontinuously may render with gaps.
- Variables with extra dimensions (e.g. vertical levels) can't be displayed
  on the map or exported to CSV; netCDF exports include them (cropped to the
  selection but otherwise as-is).
- The slice color scale is computed per time step, so colors aren't
  comparable across time steps; the legend always reflects the current
  slice.
- Value filtering without a spatial/time crop materializes the whole file
  in memory — combine filters for very large files.
