// Pairwise tab — pick two step features, compare Sim vs G4 as 2D histograms.
//
// Layout (2x2):
//   [G4 raw]          [Sim raw]
//   [G4-vs-G4 diff]   [Sim-vs-G4 diff]
//
// Bins are log10 over the same ranges the matplotlib pairwise_comparison.py
// uses. Both raw histograms are density-normalised on the backend. The two
// rel-diff panels share the LogNorm(1e-2, 1e0) jet colorscale.
import { api, setStatus } from "../api.js";
import { processesFromGlobals, processSelectionError } from "../processes.js";

const RAW_COLORSCALE  = "Viridis";
const DIFF_COLORSCALE = "Jet";
const DIFF_ZMIN_LOG   = -2;   // 1e-2
const DIFF_ZMAX_LOG   =  0;   // 1e0

function _binCenters(edges) {
  const out = new Array(edges.length - 1);
  for (let i = 0; i < out.length; i++) {
    // Geometric midpoint since bins are log-spaced.
    out[i] = Math.sqrt(edges[i] * edges[i + 1]);
  }
  return out;
}

function _transpose(z) {
  // Server returns (n_x, n_y). Plotly heatmap wants (n_y, n_x).
  const nx = z.length;
  const ny = z[0] ? z[0].length : 0;
  const out = Array.from({ length: ny }, () => new Array(nx));
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) out[j][i] = z[i][j];
  }
  return out;
}

function _log10Safe(z) {
  return z.map(row => row.map(v => (v > 0 ? Math.log10(v) : null)));
}

function _zExtent(z) {
  let zmin = Infinity, zmax = -Infinity;
  for (const row of z) for (const v of row) {
    if (v == null || !isFinite(v)) continue;
    if (v < zmin) zmin = v;
    if (v > zmax) zmax = v;
  }
  return { zmin, zmax };
}

function _logTicks(zmin, zmax) {
  const sup = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  const supStr = (n) => [...String(n)].map(c => c === "-" ? "⁻" : sup[+c]).join("");
  const tickvals = [], ticktext = [];
  for (let t = Math.floor(zmin); t <= Math.ceil(zmax); t++) {
    tickvals.push(t); ticktext.push(`10${supStr(t)}`);
  }
  return { tickvals, ticktext };
}

function _drawRaw(divId, z_nx_ny, xCenters, yCenters, title, xLabel, yLabel,
                  zmin, zmax, tickvals, ticktext) {
  const z = _log10Safe(_transpose(z_nx_ny));
  Plotly.react(divId, [{
    type: "heatmap",
    z, x: xCenters, y: yCenters,
    colorscale: RAW_COLORSCALE,
    zmin, zmax,
    colorbar: { title: "density (log₁₀)", tickvals, ticktext },
    connectgaps: false, hoverongaps: false,
    hovertemplate: `${xLabel}=%{x:.3e}<br>${yLabel}=%{y:.3e}<br>p=10^%{z:.2f}<extra></extra>`,
  }], {
    title: { text: title, font: { color: "#e6edf3" } },
    paper_bgcolor: "#161b22", plot_bgcolor: "#0e1116",
    font: { color: "#e6edf3" },
    margin: { t: 40, r: 30, b: 60, l: 70 },
    xaxis: { title: xLabel, type: "log", gridcolor: "#30363d", zeroline: false },
    yaxis: { title: yLabel, type: "log", gridcolor: "#30363d", zeroline: false },
  }, { responsive: true });
}

function _drawDiff(divId, z_nx_ny, xCenters, yCenters, title, xLabel, yLabel) {
  // Mirror plot_pairwise_relative_diff: log10 of the relative-diff matrix,
  // shared LogNorm(1e-2, 1e0) → log10 range [-2, 0]. Zero entries (masked
  // bins) are rendered transparent.
  const z = _log10Safe(_transpose(z_nx_ny));
  Plotly.react(divId, [{
    type: "heatmap",
    z, x: xCenters, y: yCenters,
    colorscale: DIFF_COLORSCALE,
    zmin: DIFF_ZMIN_LOG, zmax: DIFF_ZMAX_LOG,
    colorbar: {
      title: "rel. diff",
      tickvals: [-2, -1, 0],
      ticktext: ["10⁻²", "10⁻¹", "10⁰"],
    },
    connectgaps: false, hoverongaps: false,
    hovertemplate: `${xLabel}=%{x:.3e}<br>${yLabel}=%{y:.3e}<br>diff=10^%{z:.2f}<extra></extra>`,
  }], {
    title: { text: title, font: { color: "#e6edf3" } },
    paper_bgcolor: "#161b22", plot_bgcolor: "#0e1116",
    font: { color: "#e6edf3" },
    margin: { t: 40, r: 30, b: 60, l: 70 },
    xaxis: { title: xLabel, type: "log", gridcolor: "#30363d", zeroline: false },
    yaxis: { title: yLabel, type: "log", gridcolor: "#30363d", zeroline: false },
  }, { responsive: true });
}

export async function mountPairwise(root) {
  root.innerHTML = `
    <div class="card">
      <h2>Pairwise (Sim vs GEANT4)</h2>
      <p class="muted">
        Uses the global per-stage generators selected in the top bar.
        Pick two step features and compare 2D distributions to G4 truth.
      </p>
      <div class="controls">
        <label>G4 file</label>
        <select id="pw-file"></select>
        <label>E0 [MeV]</label>
        <input id="pw-e0" type="number" value="100" step="0.01">
        <label>n_events</label>
        <input id="pw-ne" type="number" value="1000" min="100" max="100000">
        <label>n_steps</label>
        <input id="pw-ns" type="number" value="10250" min="100" max="20000">
        <label>x feature</label>
        <select id="pw-x"></select>
        <label>y feature</label>
        <select id="pw-y"></select>
        <button id="pw-run" class="primary">Run pairwise</button>
      </div>
      <div id="pw-info" class="muted"></div>
    </div>

    <div class="row">
      <div class="col">
        <div class="card">
          <h3>GEANT4</h3>
          <div id="pw-g4" class="plot"></div>
        </div>
      </div>
      <div class="col">
        <div class="card">
          <h3>System (sim)</h3>
          <div id="pw-sim" class="plot"></div>
        </div>
      </div>
    </div>

    <div class="row">
      <div class="col">
        <div class="card">
          <h3>G4 vs G4 (split-half noise floor)</h3>
          <div id="pw-g4g4" class="plot"></div>
        </div>
      </div>
      <div class="col">
        <div class="card">
          <h3>System vs G4 (relative difference)</h3>
          <div id="pw-simg4" class="plot"></div>
        </div>
      </div>
    </div>
  `;

  // Populate G4 file dropdown.
  const fileSel = root.querySelector("#pw-file");
  for (const f of (await api.g4Files()).files) {
    const o = document.createElement("option");
    o.value = f.name; o.textContent = f.name;
    fileSel.appendChild(o);
  }
  if (fileSel.options.length) fileSel.value = fileSel.options[fileSel.options.length - 1].value;

  // Populate feature dropdowns. Defaults: x=E, y=L (the canonical pair).
  const xSel = root.querySelector("#pw-x");
  const ySel = root.querySelector("#pw-y");
  let featureLabels = {};
  try {
    const f = await api.pairwiseFeatures();
    featureLabels = f.labels || {};
    for (const name of f.features) {
      const ox = document.createElement("option");
      ox.value = name; ox.textContent = name;
      xSel.appendChild(ox);
      const oy = document.createElement("option");
      oy.value = name; oy.textContent = name;
      ySel.appendChild(oy);
    }
    xSel.value = "E";
    ySel.value = "L";
  } catch (e) {
    setStatus(`pairwise features unavailable: ${e.message}`, "error");
  }

  // Once a comparison has run, changing the x/y feature should re-run
  // automatically (no need to click "Run pairwise" again).
  let hasRun = false;

  async function runPairwise() {
    const guard = processSelectionError();
    if (guard) { setStatus(guard, "error"); return; }
    const btn  = root.querySelector("#pw-run");
    const info = root.querySelector("#pw-info");
    const oldBtn = btn.textContent;

    if (xSel.value === ySel.value) {
      info.innerHTML = `<span class="error">x and y must be different features</span>`;
      return;
    }

    btn.disabled = true;
    btn.textContent = "Running…";
    info.innerHTML = `<span class="warn">simulator running — first build of a fresh process config can take ~30–60s on CPU</span>`;
    setStatus("running pairwise comparison...");

    let stopPoll = false;
    const t0 = Date.now();
    (async () => {
      while (!stopPoll) {
        try {
          const p = await api.simProgress();
          const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
          if (p.phase === "simulating" && p.total > 0) {
            const pct = (100 * p.step / p.total).toFixed(1);
            btn.textContent = `Simulating ${pct}%`;
            info.innerHTML = `<span class="warn">simulating — ${pct}% (${p.step}/${p.total}) · ${elapsed}s</span>`;
          } else if (p.phase === "building stepping manager") {
            btn.textContent = `Building… (${elapsed}s)`;
            info.innerHTML = `<span class="warn">building stepping manager (~30–60s on CPU)</span>`;
          } else if (p.phase === "wrapping pre-built manager") {
            btn.textContent = `Preparing… (${elapsed}s)`;
            info.innerHTML = `<span class="warn">reusing pre-built stepping manager (fast path)</span>`;
          }
        } catch (e) { /* swallow */ }
        await new Promise(r => setTimeout(r, 500));
      }
    })();

    const payload = {
      g4_file:    fileSel.value,
      E0_eV:      parseFloat(root.querySelector("#pw-e0").value) * 1e6,
      n_events:   parseInt(root.querySelector("#pw-ne").value),
      n_steps:    parseInt(root.querySelector("#pw-ns").value),
      x_feature:  xSel.value,
      y_feature:  ySel.value,
      processes:  processesFromGlobals(),
    };

    try {
      const r = await api.pairwise(payload);
      const xCenters = _binCenters(r.x_bins);
      const yCenters = _binCenters(r.y_bins);
      const xLabel = featureLabels[r.x_feature] || r.x_feature;
      const yLabel = featureLabels[r.y_feature] || r.y_feature;

      // Shared color scale for the two raw histograms so they're visually
      // comparable.
      const g4Log  = _log10Safe(_transpose(r.g4_hist));
      const simLog = _log10Safe(_transpose(r.sim_hist));
      const eG4    = _zExtent(g4Log);
      const eSim   = _zExtent(simLog);
      let zmin = Math.min(eG4.zmin, eSim.zmin);
      let zmax = Math.max(eG4.zmax, eSim.zmax);
      if (!isFinite(zmin) || !isFinite(zmax) || zmin === zmax) { zmin = -6; zmax = 0; }
      const { tickvals, ticktext } = _logTicks(zmin, zmax);

      _drawRaw("pw-g4",  r.g4_hist,  xCenters, yCenters,
               `GEANT4 (N=${r.n_g4})`,  xLabel, yLabel, zmin, zmax, tickvals, ticktext);
      _drawRaw("pw-sim", r.sim_hist, xCenters, yCenters,
               `System (N=${r.n_sim})`, xLabel, yLabel, zmin, zmax, tickvals, ticktext);
      _drawDiff("pw-g4g4",  r.g4_vs_g4,  xCenters, yCenters,
                "G4 vs G4 (split-half)", xLabel, yLabel);
      _drawDiff("pw-simg4", r.sim_vs_g4, xCenters, yCenters,
                "System vs G4", xLabel, yLabel);

      info.innerHTML = `<span class="good">done · N_g4=${r.n_g4}, N_sim=${r.n_sim}</span>`;
      setStatus(`pairwise done · ${r.x_feature} × ${r.y_feature}`);
      hasRun = true;
    } catch (e) {
      console.error("[pairwise] failed", e);
      info.innerHTML = `<span class="error">${e.message}</span>`;
      setStatus(`pairwise failed: ${e.message}`, "error");
    } finally {
      stopPoll = true;
      btn.disabled = false;
      btn.textContent = oldBtn;
    }
  }

  root.querySelector("#pw-run").addEventListener("click", runPairwise);

  // Auto re-run when the x/y feature changes, but only after an initial run
  // (so the page doesn't fire a simulation on first load). Guard against the
  // degenerate x===y case, which runPairwise() reports as an error.
  const onFeatureChange = () => {
    if (hasRun && xSel.value !== ySel.value) runPairwise();
  };
  xSel.addEventListener("change", onFeatureChange);
  ySel.addEventListener("change", onFeatureChange);
}
