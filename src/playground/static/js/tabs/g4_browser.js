// G4 data browser tab.
// 2D phase-space histogram (KE × StepLength by default) on log color scale.
// Clicking pins a phase-space point; depending on the global selector mode the
// 2D plot overlays either the chosen bin (rectangle) or the chosen point
// (marker). "Show regions" toggles analytical phase-space region boundaries.
import { api, setStatus } from "../api.js";
import {
  setPhaseSpacePoint, getPhaseSpacePoint, onChange,
  getSelectorMode, getHalfwidthDex,
} from "../state.js";

export async function mountG4Browser(root) {
  root.innerHTML = `
    <div class="card">
      <h2>GEANT4 phase space (log color)</h2>
      <div class="controls">
        <label>file</label>
        <select id="g4-file"></select>
        <label>x</label>
        <select id="g4-x">
          <option value="KineticEnergy" selected>KineticEnergy</option>
          <option value="StepLength">StepLength</option>
          <option value="ContinuousLoss">ContinuousLoss</option>
          <option value="SecondaryLoss">SecondaryLoss</option>
        </select>
        <label>y</label>
        <select id="g4-y">
          <option value="StepLength" selected>StepLength</option>
          <option value="ContinuousLoss">ContinuousLoss</option>
          <option value="SecondaryLoss">SecondaryLoss</option>
        </select>
        <label>bins</label>
        <input id="g4-bins" type="number" value="120" min="20" max="500">
        <button id="g4-reload" class="primary">Reload</button>
        <label style="margin-left:8px"><input type="checkbox" id="g4-show-regions"> Show regions</label>
      </div>
      <div id="g4-2d" class="plot plot-tall"></div>
      <p class="muted">Click a cell to pin a phase-space point. Mode/halfwidth are global (top bar).</p>
    </div>
  `;

  const fileSel = root.querySelector("#g4-file");
  const xSel    = root.querySelector("#g4-x");
  const ySel    = root.querySelector("#g4-y");
  const binsIn  = root.querySelector("#g4-bins");

  const files = (await api.g4Files()).files;
  for (const f of files) {
    const o = document.createElement("option");
    o.value = f.name; o.textContent = `${f.name} (${f.ext.join(", ")})`;
    fileSel.appendChild(o);
  }
  if (files.length > 1) fileSel.value = files[files.length - 1].name;

  let lastH = null;
  // The plotly_click handler is attached exactly once (see _wireClick). Plotly's
  // gd.on() is additive and Plotly.react() does NOT purge user listeners, so
  // re-attaching inside drawHist2d would accumulate one handler per redraw and
  // each click would fire them all — a geometric blow-up that grinds the tab to
  // a halt after a few clicks.
  let clickWired = false;
  let regionsCache = null;

  // ─── Regions helpers ──────────────────────────────────────────────────────
  function _hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function _buildBoundaryTraces(regions, keysAsc, rotations) {
    // Returns {fillTraces, lineTraces, labels} to layer around the heatmap.
    // regions: full API response from /phase_space_regions.
    // keysAsc: boundary keys in ascending-L order; N keys → N+1 bands (regions).
    // rotations: per-region label tilt in matplotlib (CCW-positive) degrees.
    // labels are Plotly layout annotations (NOT traces) — they live in layout.
    if (!lastH) return { fillTraces: [], lineTraces: [], labels: [] };

    const E   = regions.energy_eV;
    const bnd = regions.boundaries;

    const yMin = lastH.y_edges[0];
    const yMax = lastH.y_edges[lastH.y_edges.length - 1];

    const curves = keysAsc.map(k => bnd[k]);  // N ascending-L boundary curves
    const N = curves.length;

    // Lower bounds: region 0 starts at yMin, region k>=1 starts at curve[k-1]
    // (raw — a null lower makes the fill guard skip that region at that energy).
    const lowerBounds = [E.map(() => yMin), ...curves];

    // Upper bounds with a cascading fallback that mirrors the r2_upper/r3_upper/
    // tl_or_max cascade in benchmark/physics_only/phase_space_pdfs.py: when an
    // interior boundary is undefined at some energy, the simpler region below it
    // absorbs the space up to the next *existing* boundary above (or yMax for the
    // open-topped final region) instead of leaving a gap. Walk regions top-down so
    // each region inherits the effective upper of the region above where its own
    // boundary is null. This keeps the continuous overlay from exposing bare
    // heatmap where ion_gauss/exc_freq vanish at low E.
    const upperBounds = new Array(N + 1);
    upperBounds[N] = E.map(() => yMax);
    for (let k = N - 1; k >= 0; k--) {
      const curve = curves[k];
      const above = upperBounds[k + 1];
      upperBounds[k] = curve.map((v, i) => (v === null ? above[i] : v));
    }

    const fillTraces = regions.regions.map((reg, idx) => {
      const lower = lowerBounds[idx];
      const upper = upperBounds[idx];
      const xFwd = [], yFwd = [], xRev = [], yRev = [];
      for (let i = 0; i < E.length; i++) {
        const lo = lower[i], hi = upper[i];
        if (lo !== null && hi !== null && lo > 0 && hi > 0) {
          xFwd.push(E[i]); yFwd.push(hi);
          xRev.unshift(E[i]); yRev.unshift(lo);
        }
      }
      if (xFwd.length === 0) return null;
      return {
        type: "scatter", mode: "none",
        x: [...xFwd, ...xRev, xFwd[0]],
        y: [...yFwd, ...yRev, yFwd[0]],
        fill: "toself",
        fillcolor: _hexToRgba(reg.color, reg.alpha),
        line: { width: 0 },
        showlegend: false, hoverinfo: "skip",
      };
    }).filter(Boolean);

    const lineTraces = keysAsc.map(key => ({
      type: "scatter", mode: "lines",
      x: E.filter((_, i) => bnd[key][i] !== null),
      y: bnd[key].filter(v => v !== null),
      line: { color: "black", width: 1 },
      showlegend: false, hoverinfo: "skip",
    }));

    // In-region text labels (matplotlib-style), mirroring the ax.text placement
    // in benchmark/physics_only/phase_space_pdfs.py: anchor at the middle valid
    // energy, vertically at the band's geometric-mean center, tilted per region.
    // Rendered as layout annotations so they get a rotation (textangle) and a
    // semi-transparent white bbox for readability over the heatmap. NB: unlike
    // the raw-valued line/fill traces, annotation coords follow the axis type —
    // log10(value) on a log axis, raw value on a linear one.
    const xCoord = (v) => (lastH.log_x ? Math.log10(v) : v);
    const yCoord = (v) => (lastH.log_y ? Math.log10(v) : v);
    const labels = regions.regions.map((reg, idx) => {
      const lower = lowerBounds[idx], upper = upperBounds[idx];
      const valid = [];
      for (let i = 0; i < E.length; i++) {
        const lo = lower[i], hi = upper[i];
        if (lo !== null && hi !== null && lo > 0 && hi > 0) valid.push(i);
      }
      if (valid.length === 0) return null;
      const i = valid[Math.floor(valid.length / 2)];
      const yMid = Math.sqrt(lower[i] * upper[i]);   // geometric-mean band center
      // Skip labels whose center is off the visible plot — Plotly layout
      // annotations aren't clipped to the axes and would otherwise spill into
      // the margin.
      if (E[i] < lastH.x_edges[0] || E[i] > lastH.x_edges[lastH.x_edges.length - 1] ||
          yMid < yMin || yMid > yMax) return null;
      return {
        xref: "x", yref: "y",
        x: xCoord(E[i]),
        y: yCoord(yMid),
        text: reg.label.replace(/\n/g, "<br>"),
        showarrow: false, align: "center",
        textangle: -(rotations?.[idx] ?? 0),                       // matplotlib CCW → plotly CW
        font: { color: "#000", size: 11 },
        bgcolor: "rgba(255,255,255,0.6)", bordercolor: "rgba(0,0,0,0)",
      };
    }).filter(Boolean);

    return { fillTraces, lineTraces, labels };
  }

  // Boundary keys in ascending-L order. N keys → N+1 bands (regions).
  const _CNT_KEYS = ["very_thin", "ion_gauss", "exc_freq", "thick_limit"];
  // Per-region label tilt (matplotlib CCW-positive degrees), matching
  // region_rotations in phase_space_pdfs.py.
  const _CNT_ROTATIONS = [0, 20, 20, 0, 90];

  // ─── 2D histogram draw ────────────────────────────────────────────────────
  async function reload2d() {
    // No track files (e.g. a fresh checkout whose gitignored storage/ has no
    // GEANT4 tracks): skip the request rather than send file= and get a 400.
    if (!fileSel.value) {
      setStatus("no GEANT4 track files found in storage/.../tracks", "warn");
      return;
    }
    setStatus("loading G4 2D histogram...");
    const h = await api.g4Hist2d({
      file: fileSel.value, x: xSel.value, y: ySel.value,
      bins: binsIn.value, log_x: 1, log_y: 1,
    });
    lastH = h;
    drawHist2d(h);
    setStatus(`G4 2D ${h.x_col} × ${h.y_col}`);
  }

  function _zLog(z) {
    // log10(count + 1) gives zero for empty cells, so they render as the
    // lowest color rather than NaN. Plotly draws NaN cells as transparent.
    return z.map(c => Math.log10(c + 1));
  }

  function _overlayShapesAndTraces() {
    const onEL = lastH &&
      lastH.x_col === "KineticEnergy" && lastH.y_col === "StepLength";
    const cntEl = root.querySelector("#g4-show-regions");
    const showCnt = cntEl && !cntEl.disabled && cntEl.checked && regionsCache && onEL;

    let regionFillTraces = [], regionLineTraces = [], regionLabels = [];
    if (showCnt) {
      const b = _buildBoundaryTraces(regionsCache, _CNT_KEYS, _CNT_ROTATIONS);
      regionFillTraces.push(...b.fillTraces);
      regionLineTraces.push(...b.lineTraces);
      regionLabels.push(...b.labels);
    }

    if (!lastH) return {
      shapes: [], annotations: regionLabels,
      extraTraces: [...regionFillTraces, ...regionLineTraces],
    };
    if (lastH.x_col !== "KineticEnergy" || lastH.y_col !== "StepLength")
      return { shapes: [], annotations: [], extraTraces: [] };

    const { E_eV, L_m } = getPhaseSpacePoint();
    if (E_eV == null || L_m == null)
      return {
        shapes: [], annotations: regionLabels,
        extraTraces: [...regionFillTraces, ...regionLineTraces],
      };

    const mode = getSelectorMode();
    if (mode === "bin") {
      const hw = getHalfwidthDex();
      const lo = (v) => Math.pow(10, Math.log10(v) - hw);
      const hi = (v) => Math.pow(10, Math.log10(v) + hw);
      return {
        shapes: [{
          type: "rect", xref: "x", yref: "y",
          x0: lo(E_eV), x1: hi(E_eV),
          y0: lo(L_m),  y1: hi(L_m),
          line: { color: "#f85149", width: 2 },
          fillcolor: "rgba(248,81,73,0.10)",
        }],
        annotations: regionLabels,
        extraTraces: [...regionFillTraces, ...regionLineTraces],
      };
    }
    // mode === "point"
    return {
      shapes: [], annotations: regionLabels,
      extraTraces: [
        ...regionFillTraces,
        ...regionLineTraces,
        {
          type: "scatter", mode: "markers", x: [E_eV], y: [L_m],
          marker: { color: "#f85149", size: 12, symbol: "x", line: { width: 2 } },
          name: "selected", showlegend: false, hoverinfo: "skip",
        },
      ],
    };
  }

  function drawHist2d(h) {
    const xc = h.x_edges.slice(0, -1).map((e, i) => 0.5 * (e + h.x_edges[i+1]));
    const yc = h.y_edges.slice(0, -1).map((e, i) => 0.5 * (e + h.y_edges[i+1]));
    const zRaw = yc.map((_, j) => xc.map((_, i) => Math.max(h.z[i][j], 0)));
    const zLog = zRaw.map(_zLog);
    // Build colorbar ticks that read as integer counts (1, 10, 100, …).
    const maxZ = Math.max(...zRaw.flat());
    const maxL = Math.log10(maxZ + 1);
    const tickPows = [];
    for (let p = 0; Math.pow(10, p) - 1 <= maxZ; p++) tickPows.push(p);
    const tickvals = tickPows.map(p => Math.log10(Math.pow(10, p) + 1) - 0); // = p
    const ticktext = tickPows.map(p => Math.pow(10, p).toString());

    const { shapes, extraTraces, annotations } = _overlayShapesAndTraces();

    const fillTraces  = extraTraces.filter(t => t.fill === "toself");
    const otherTraces = extraTraces.filter(t => t.fill !== "toself");

    Plotly.react("g4-2d", [
      ...fillTraces,
      {
        type: "heatmap", x: xc, y: yc, z: zLog, colorscale: "Viridis",
        hovertemplate: `${h.x_col}: %{x:.3e}<br>${h.y_col}: %{y:.3e}` +
                       `<br>log10(N+1): %{z:.2f}<extra></extra>`,
        colorbar: { title: "counts (log)", tickvals, ticktext },
        zmin: 0, zmax: maxL,
      },
      ...otherTraces,
    ], {
      paper_bgcolor: "#161b22", plot_bgcolor: "#0e1116",
      font: { color: "#e6edf3" }, margin: { t: 10, r: 30, b: 50, l: 70 },
      xaxis: { title: h.x_col, type: h.log_x ? "log" : "linear", gridcolor: "#30363d" },
      yaxis: { title: h.y_col, type: h.log_y ? "log" : "linear", gridcolor: "#30363d" },
      shapes,
      annotations,
    }, { responsive: true });

    _wireClick();
    _updateRegionsCheckbox();
  }

  // Attach the plotly_click handler once. The handler reads the module-scoped
  // `lastH` (always the latest histogram) rather than a captured `h`, and it
  // does NOT redraw directly: setPhaseSpacePoint() fires the global onChange
  // subscriber below, which re-draws the bin/point overlay. Wiring this inside
  // drawHist2d (as before) leaked one listener per redraw.
  function _wireClick() {
    if (clickWired) return;
    clickWired = true;
    document.getElementById("g4-2d").on("plotly_click", ev => {
      const pt = ev.points[0];
      if (lastH && lastH.x_col === "KineticEnergy" && lastH.y_col === "StepLength") {
        setPhaseSpacePoint(pt.x, pt.y);
      } else {
        setStatus(`clicked (${pt.x.toExponential(3)}, ${pt.y.toExponential(3)}) — ` +
                  `not (E, L), point not pinned`, "warn");
      }
    });
  }

  function _updateRegionsCheckbox() {
    const active = lastH &&
                   lastH.x_col === "KineticEnergy" &&
                   lastH.y_col === "StepLength";
    for (const id of ["#g4-show-regions"]) {
      const el = root.querySelector(id);
      if (!el) continue;
      el.disabled = !active;
      if (!active) el.checked = false;
    }
  }

  // ─── Event wiring ─────────────────────────────────────────────────────────
  root.querySelector("#g4-reload").addEventListener("click", reload2d);
  fileSel.addEventListener("change", reload2d);
  xSel.addEventListener("change", reload2d);
  ySel.addEventListener("change", reload2d);

  root.querySelector("#g4-show-regions").addEventListener("change", async () => {
    const el = root.querySelector("#g4-show-regions");
    if (el.checked && regionsCache === null) {
      setStatus("loading phase-space region boundaries...");
      try {
        regionsCache = await api.g4PhaseSpaceRegions();
        setStatus("phase-space regions loaded");
      } catch (e) {
        setStatus(`failed to load region boundaries: ${e.message}`, "error");
        el.checked = false;
      }
    }
    if (lastH) drawHist2d(lastH);
  });

  // React to global state changes: mode, halfwidth.
  onChange(() => {
    if (lastH) drawHist2d(lastH);
  });

  await reload2d();
}
