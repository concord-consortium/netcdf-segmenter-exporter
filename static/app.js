let metadata = null;
let overlay = null;
let drawnShape = null; // {type: "bbox", value: {...}} | {type: "polygon", value: [[lon,lat],...]}
let overlayRequestSeq = 0;

const map = L.map("map").setView([20, 0], 2);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  noWrap: true,
  bounds: [[-90, -180], [90, 180]],
}).addTo(map);

const drawnItems = new L.FeatureGroup().addTo(map);
map.addControl(new L.Control.Draw({
  draw: {
    rectangle: true,
    polygon: true,
    polyline: false,
    circle: false,
    marker: false,
    circlemarker: false,
  },
  edit: { featureGroup: drawnItems, edit: false, remove: false },
}));

// wrap any longitude into [-180, 180] (leaflet world-copies can exceed it);
// +180 stays +180 so an edge-of-map selection isn't mistaken for a crossing
function wrapLon(lon) {
  if (lon === 180) return 180;
  return ((lon + 180) % 360 + 360) % 360 - 180;
}

map.on(L.Draw.Event.CREATED, (e) => {
  drawnItems.clearLayers(); // one shape at a time
  drawnItems.addLayer(e.layer);
  if (e.layerType === "rectangle") {
    const b = e.layer.getBounds();
    const value = {
      west: wrapLon(b.getWest()), south: b.getSouth(),
      east: wrapLon(b.getEast()), north: b.getNorth(),
    };
    if (value.east <= value.west) {
      drawnItems.clearLayers();
      drawnShape = null;
      setText("shape-info",
        "Shapes crossing the antimeridian aren't supported — draw within one world copy.");
      return;
    }
    drawnShape = { type: "bbox", value };
    setText("shape-info", "Bounding box selected.");
  } else {
    const ring = e.layer.toGeoJSON().geometry.coordinates[0]
      .map(([lon, lat]) => [wrapLon(lon), lat]);
    const lons = ring.map((p) => p[0]);
    if (Math.max(...lons) - Math.min(...lons) > 180) {
      drawnItems.clearLayers();
      drawnShape = null;
      setText("shape-info",
        "Shapes crossing the antimeridian aren't supported — draw within one world copy.");
      return;
    }
    drawnShape = { type: "polygon", value: ring };
    setText("shape-info", "Polygon selected.");
  }
});

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

async function openFile() {
  const path = document.getElementById("file-path").value.trim();
  if (!path) return;
  try {
    setText("status", "Opening…");
    const res = await fetch("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      setText("status", `Error: ${err.detail}`);
      return;
    }
    metadata = await res.json();
    setText("status", "");
    applyMetadata();
    await refreshOverlay();
  } catch (err) {
    setText("status", `Request failed: ${err.message}`);
  }
}

function applyMetadata() {
  stopPlayback();
  const sizeMb = (metadata.size_bytes / 1048576).toFixed(1);
  setText("file-info", `${metadata.path} (${sizeMb} MB)`);

  const varSel = document.getElementById("variable-select");
  const filterSel = document.getElementById("filter-variable");
  varSel.innerHTML = "";
  filterSel.innerHTML = '<option value="">(none)</option>';
  for (const v of metadata.variables) {
    const label = v.units ? `${v.name} (${v.units})` : v.name;
    varSel.add(new Option(label, v.name));
    filterSel.add(new Option(label, v.name));
  }

  const slider = document.getElementById("time-slider");
  const count = metadata.time ? metadata.time.count : 1;
  slider.max = String(Math.max(0, count - 1));
  slider.value = "0";
  updateTimeLabel();

  if (metadata.time) {
    document.getElementById("time-start").value = metadata.time.start;
    document.getElementById("time-end").value = metadata.time.end;
  } else {
    document.getElementById("time-start").value = "";
    document.getElementById("time-end").value = "";
  }

  const ext = metadata.edges;
  map.fitBounds([[ext.south, ext.west], [ext.north, ext.east]]);
}

function updateTimeLabel() {
  const idx = Number(document.getElementById("time-slider").value);
  if (metadata && metadata.time && metadata.time.values) {
    setText("time-label", metadata.time.values[idx]);
  } else if (metadata && metadata.time) {
    setText("time-label", `index ${idx} of ${metadata.time.count - 1}`);
  } else {
    setText("time-label", "(no time axis)");
  }
}

let playing = false;
let playToken = 0;

function timeStepCount() {
  return metadata && metadata.time ? metadata.time.count : 0;
}

function setTimeIndex(idx) {
  document.getElementById("time-slider").value = String(idx);
  updateTimeLabel();
}

async function nudge(delta) {
  const count = timeStepCount();
  if (count <= 1) return;
  const slider = document.getElementById("time-slider");
  const idx = Math.min(count - 1, Math.max(0, Number(slider.value) + delta));
  if (idx === Number(slider.value)) return;
  setTimeIndex(idx);
  await refreshOverlay();
}

function stopPlayback() {
  playing = false;
  playToken += 1; // cancels any in-flight play loop
  const btn = document.getElementById("play-btn");
  btn.textContent = "▶ Play";
  btn.setAttribute("aria-label", "Play");
}

async function startPlayback() {
  const count = timeStepCount();
  if (count <= 1) return;
  const token = ++playToken;
  playing = true;
  const btn = document.getElementById("play-btn");
  btn.textContent = "⏸ Pause";
  btn.setAttribute("aria-label", "Pause");

  const slider = document.getElementById("time-slider");
  if (Number(slider.value) >= count - 1) {
    setTimeIndex(0); // play pressed at the end: restart from the start
    const ok = await refreshOverlay();
    if (token !== playToken) return;
    if (!ok) {
      stopPlayback();
      return;
    }
  }

  while (token === playToken) {
    const started = performance.now();
    const idx = Number(slider.value);
    if (idx >= count - 1) break; // reached the end
    setTimeIndex(idx + 1);
    const ok = await refreshOverlay();
    if (token !== playToken) return; // paused or restarted while rendering
    if (!ok) break; // genuine render failure: stop, don't error-loop
    const stepsPerSec = Number(document.getElementById("speed-select").value);
    const dwell = Math.max(0, 1000 / stepsPerSec - (performance.now() - started));
    await new Promise((resolve) => setTimeout(resolve, dwell));
  }
  if (token === playToken) stopPlayback();
}

function togglePlayback() {
  if (playing) stopPlayback();
  else startPlayback();
}

async function refreshOverlay() {
  if (!metadata) return false;
  const variable = document.getElementById("variable-select").value;
  if (!variable) return false;
  const idx = document.getElementById("time-slider").value;
  const seq = ++overlayRequestSeq;
  try {
    const res = await fetch(
      `/api/slice?variable=${encodeURIComponent(variable)}&time_index=${idx}`
    );
    if (seq !== overlayRequestSeq) return true; // a newer request owns the overlay — not a failure
    if (!res.ok) {
      setText("status", "Failed to render slice.");
      return false;
    }
    setText("legend-min", Number(res.headers.get("X-Vmin")).toPrecision(4));
    setText("legend-max", Number(res.headers.get("X-Vmax")).toPrecision(4));
    const blob = await res.blob();
    if (seq !== overlayRequestSeq) return true; // a newer request owns the overlay — not a failure
    const url = URL.createObjectURL(blob);
    const ext = metadata.edges;
    const bounds = [[ext.south, ext.west], [ext.north, ext.east]];
    if (overlay) {
      const oldUrl = overlay._url;
      overlay.setUrl(url);
      overlay.setBounds(L.latLngBounds(bounds));
      if (oldUrl && oldUrl.startsWith("blob:")) URL.revokeObjectURL(oldUrl);
    } else {
      overlay = L.imageOverlay(url, bounds, {
        opacity: 0.75,
        className: "data-overlay", // crisp cells for coarse grids (style.css)
      }).addTo(map);
    }
    return true;
  } catch (err) {
    setText("status", "Failed to render slice.");
    return false;
  }
}

function buildFilters() {
  const filters = {};
  if (drawnShape && drawnShape.type === "bbox") filters.bbox = drawnShape.value;
  if (drawnShape && drawnShape.type === "polygon") filters.polygon = drawnShape.value;

  const start = document.getElementById("time-start").value.trim();
  const end = document.getElementById("time-end").value.trim();
  if (start || end) filters.time_range = { start: start || null, end: end || null };

  const fvar = document.getElementById("filter-variable").value;
  if (fvar) {
    const min = document.getElementById("filter-min").value;
    const max = document.getElementById("filter-max").value;
    filters.var_filter = {
      variable: fvar,
      min: min === "" ? null : Number(min),
      max: max === "" ? null : Number(max),
    };
  }
  return filters;
}

async function exportSubset(format) {
  if (!metadata) {
    setText("status", "Open a file first.");
    return;
  }
  setText("status", "Preparing export…");
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format, ...buildFilters() }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      const detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
      setText("status", `Export failed: ${detail}`);
      return;
    }
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const cd = res.headers.get("Content-Disposition") || "";
    const match = cd.match(/filename="(.+)"/);
    a.download = match ? match[1] : (format === "csv" ? "subset.csv" : "subset.nc");
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    setText("status", "Export downloaded.");
  } catch (err) {
    setText("status", `Request failed: ${err.message}`);
  }
}

document.getElementById("open-btn").addEventListener("click", openFile);
document.getElementById("file-path").addEventListener("keydown", (e) => {
  if (e.key === "Enter") openFile();
});
document.getElementById("variable-select").addEventListener("change", refreshOverlay);
document.getElementById("time-slider").addEventListener("input", () => {
  if (playing) stopPlayback(); // user takes control
  updateTimeLabel();
});
document.getElementById("time-slider").addEventListener("change", refreshOverlay);
document.getElementById("clear-shape-btn").addEventListener("click", () => {
  drawnItems.clearLayers();
  drawnShape = null;
  setText("shape-info", "No shape drawn (whole globe).");
});
document.getElementById("export-csv-btn").addEventListener("click", () => exportSubset("csv"));
document.getElementById("export-nc-btn").addEventListener("click", () => exportSubset("netcdf"));
document.getElementById("play-btn").addEventListener("click", togglePlayback);
document.getElementById("step-back-btn").addEventListener("click", () => nudge(-1));
document.getElementById("step-fwd-btn").addEventListener("click", () => nudge(1));
