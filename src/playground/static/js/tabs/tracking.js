// Trajectory + Pull tabs.
// Both tabs build the stepping-manager `processes` list from the global
// per-stage generator dropdowns in the top bar; there is no per-tab
// preset / custom-JSON / use-globals toggle.
import { api, setStatus } from "../api.js";
import { processesFromGlobals as _processesFromGlobals, processSelectionError } from "../processes.js";


// ─────────────────────────────────────────────────────────────────────────
// Trajectory tab — run the simulator and render 3D trajectories + feature summary.
// ─────────────────────────────────────────────────────────────────────────
export async function mountTrajectory(root) {
  root.innerHTML = `
    <div class="card">
      <h2>Run trajectory simulator</h2>
      <p class="muted">Uses the global per-stage generators selected in the top bar.</p>
      <div class="controls">
        <label>E0 [MeV]</label>
        <input id="ts-e0" type="number" value="100" step="0.01">
        <label>n_events</label>
        <input id="ts-ne" type="number" value="10" min="1" max="50000">
        <label>n_steps</label>
        <input id="ts-ns" type="number" value="10250" min="1" max="20000">
        <label>trajectories shown</label>
        <input id="ts-sub" type="number" value="25" min="1" max="200">
        <button id="ts-run" class="primary">Run simulator</button>
      </div>
      <div id="ts-info" class="muted"></div>
    </div>

    <div class="row">
      <div class="col">
        <div class="card">
          <h2>Sample trajectories (3D)</h2>
          <div id="ts-3d" class="plot plot-tall"></div>
        </div>
      </div>
      <div class="col">
        <div class="card">
          <h2>Step features summary</h2>
          <div id="ts-feats"></div>
        </div>
      </div>
    </div>
  `;

  async function runSim() {
    const guard = processSelectionError();
    if (guard) { setStatus(guard, "error"); return; }
    const btn  = root.querySelector("#ts-run");
    const info = root.querySelector("#ts-info");
    const oldBtnText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Running… (first run can take 30–60 s)";
    info.innerHTML = `<span class="warn">simulator running — please wait, the first build of a fresh process config can take a minute on CPU</span>`;
    setStatus("running trajectory simulator (may take 30–60s on first call)...");

    let stopPolling = false;
    const t0 = Date.now();
    const fmt = (n) => (n >= 1000 ? n.toLocaleString() : String(n));
    const pollProgress = async () => {
      while (!stopPolling) {
        try {
          const p = await api.simProgress();
          const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
          if (p.phase === "simulating" && p.total > 0) {
            const pct = (100 * p.step / p.total).toFixed(1);
            const rate = p.step > 0 ? (p.step / parseFloat(elapsed)).toFixed(1) : "—";
            const etaSec = (p.step > 0 && rate !== "—")
              ? ((p.total - p.step) / parseFloat(rate)).toFixed(0)
              : "?";
            btn.textContent = `Simulating ${pct}% (${fmt(p.step)}/${fmt(p.total)})`;
            info.innerHTML =
              `<span class="warn">simulating — ${pct}% · step ${fmt(p.step)}/${fmt(p.total)} · ${rate} steps/s · elapsed ${elapsed}s · ETA ~${etaSec}s</span>`;
          } else if (p.phase === "building stepping manager") {
            btn.textContent = `Building stepping manager… (${elapsed}s)`;
            info.innerHTML = `<span class="warn">building stepping manager — first call on a fresh process config can take ~30–60 s on CPU</span>`;
          } else if (p.phase === "wrapping pre-built manager") {
            btn.textContent = `Preparing simulator… (${elapsed}s)`;
            info.innerHTML = `<span class="warn">reusing pre-built stepping manager (fast path)</span>`;
          }
        } catch (e) { /* swallow */ }
        await new Promise(r => setTimeout(r, 500));
      }
    };
    pollProgress();

    const payload = {
      E0_eV: parseFloat(root.querySelector("#ts-e0").value) * 1e6,
      n_events: root.querySelector("#ts-ne").value,
      n_steps:  root.querySelector("#ts-ns").value,
      trajectory_subsample: root.querySelector("#ts-sub").value,
      processes: _processesFromGlobals(),
    };

    console.log("[trajectory] POST /api/stepping/run", payload);
    try {
      const r = await api.runSim(payload);
      console.log("[trajectory] response", { n_trajectories_returned: r.n_trajectories_returned, n_events: r.n_events });
      if (!r.trajectories || r.trajectories.length === 0) {
        info.innerHTML = `<span class="warn">simulator returned 0 trajectories (n_events=${r.n_events}, n_steps=${r.n_steps}). Check that E0 ≫ E_kill (1 keV) and that the process config is sane.</span>`;
        Plotly.purge("ts-3d");
      } else {
        draw3D(r.trajectories);
        info.textContent =
          `${r.n_trajectories_returned} of ${r.n_events} trajectories plotted, ${r.n_steps} steps max`;
      }
      drawFeats(r.feature_summary);
      setStatus(`sim done: ${r.n_events} events × ${r.n_steps} steps`);
    } catch (e) {
      console.error("[trajectory] sim failed", e);
      setStatus(`sim failed: ${e.message}`, "error");
      info.innerHTML = `<span class="error">${e.message}</span>`;
    } finally {
      stopPolling = true;
      btn.disabled = false;
      btn.textContent = oldBtnText;
    }
  }

  function draw3D(trajectories) {
    // Match the GEANT4-style colouring used in interactive_track_viewer.py:
    // jet colormap mapping kinetic energy → colour, with E0 capping the scale
    // so that trajectories fade from red (high E) to blue (low E) as the proton
    // slows down. Plotly's scatter3d `line.color` only accepts a scalar, so
    // the energy gradient is rendered via marker.color (one trace per trajectory)
    // with a faint line beneath linking the points.
    const goodTrajectories = trajectories.filter(t => t.x && t.x.length >= 2);
    console.log(`[trajectory] draw3D: ${goodTrajectories.length}/${trajectories.length} trajectories with >=2 points`);
    // eV → MeV; cap at the max KE observed across all trajectories (≈ E0).
    let eMaxMeV = 0;
    for (const t of goodTrajectories) {
      if (t.ke?.length) {
        for (const e of t.ke) {
          const m = e * 1e-6;
          if (m > eMaxMeV) eMaxMeV = m;
        }
      }
    }
    if (eMaxMeV <= 0) eMaxMeV = 1;

    const traces = goodTrajectories.map((t, idx) => {
      const keMeV = (t.ke || []).map(e => e * 1e-6);
      return {
        type: "scatter3d", mode: "lines+markers", x: t.x, y: t.y, z: t.z,
        // Subtle background line so the trajectory is visible between markers.
        line: { width: 2, color: "rgba(200,210,230,0.25)" },
        marker: {
          size: 2.2,
          color: keMeV,
          colorscale: "Jet",
          cmin: 0, cmax: eMaxMeV,
          showscale: idx === 0,
          colorbar: idx === 0 ? {
            title: "KE [MeV]", thickness: 12, len: 0.7,
            tickfont: { color: "#e6edf3" },
            titlefont: { color: "#e6edf3" },
          } : undefined,
        },
        name: `ev ${t.event}`, showlegend: false, hoverinfo: "skip",
      };
    });
    Plotly.react("ts-3d", traces, {
      paper_bgcolor: "#1a1a2e", plot_bgcolor: "#1a1a2e",
      font: { color: "#e0e0e0" }, margin: { t: 10, r: 10, b: 10, l: 10 },
      scene: {
        xaxis: { title: "x [m]", gridcolor: "#2a2a4a", backgroundcolor: "#1a1a2e",
                 color: "#e0e0e0", zerolinecolor: "#2a2a4a" },
        yaxis: { title: "y [m]", gridcolor: "#2a2a4a", backgroundcolor: "#1a1a2e",
                 color: "#e0e0e0", zerolinecolor: "#2a2a4a" },
        zaxis: { title: "z [m]", gridcolor: "#2a2a4a", backgroundcolor: "#1a1a2e",
                 color: "#e0e0e0", zerolinecolor: "#2a2a4a" },
      },
    }, { responsive: true });
  }

  function drawFeats(summary) {
    const el = root.querySelector("#ts-feats");
    if (!summary || !Object.keys(summary).length) { el.innerHTML = "<p class='muted'>no data</p>"; return; }
    const rows = Object.entries(summary).map(([k, v]) =>
      `<tr><td>${k}</td><td>${Number(v.mean).toExponential(3)}</td><td>${Number(v.std).toExponential(3)}</td></tr>`).join("");
    el.innerHTML = `
      <table style="width:100%;border-collapse:collapse;font-family:ui-monospace,monospace;font-size:12px">
        <thead><tr style="color:#8b949e;text-align:left"><th>feature</th><th>mean</th><th>std</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  root.querySelector("#ts-run").addEventListener("click", runSim);
}

// ─────────────────────────────────────────────────────────────────────────
// Pull tab — 1500×1500 pull analysis vs G4.
// ─────────────────────────────────────────────────────────────────────────
export async function mountPull(root) {
  root.innerHTML = `
    <div class="card">
      <h2 id="pull-heading">Pull analysis vs GEANT4 (1500 × 1500)</h2>
      <p class="muted">Uses the global per-stage generators selected in the top bar.</p>
      <div class="controls">
        <label>E0 [MeV]</label>
        <input id="pl-e0" type="number" value="100" step="0.01">
        <label>G4 file</label>
        <select id="pull-file"></select>
        <label>n_events</label>
        <input id="pull-ne" type="number" value="10000" min="100" max="100000">
        <label>n_steps</label>
        <input id="pull-ns" type="number" value="10250" min="100" max="20000">
        <label>min counts/bin</label>
        <input id="pull-min" type="number" value="3" min="1" max="50">
        <label>bins x</label>
        <input id="pull-binsx" type="number" value="1500" min="50" max="3000" step="50">
        <label>bins y</label>
        <input id="pull-binsy" type="number" value="1500" min="50" max="3000" step="50">
        <label>bins 1D</label>
        <input id="pull-bins1d" type="number" value="1000" min="10" max="10000" step="10">
        <button id="pull-run" class="primary">Run pull analysis</button>
        <button id="pull-rebin" disabled>Re-bin</button>
      </div>
      <div class="row">
        <div class="col">
          <h3 id="pull-sim-title">System</h3>
          <div id="pull-sim-map" class="plot"></div>
        </div>
        <div class="col">
          <h3>GEANT4</h3>
          <div id="pull-g4-map" class="plot"></div>
        </div>
      </div>
      <div class="row">
        <div class="col">
          <h3>System vs GEANT4 pull map</h3>
          <div id="pull-map" class="plot"></div>
        </div>
        <div class="col">
          <h3>System vs GEANT4 pull</h3>
          <div id="pull-hist" class="plot"></div>
        </div>
      </div>
      <div id="pull-meta" class="kv"></div>
    </div>
  `;

  const pullFileSel = root.querySelector("#pull-file");

  // Sim-defining params of the last successful full run; Re-bin reuses these
  // so it never changes what was simulated, only how it is binned.
  let lastRunSimParams = null;

  const readBins = () => [
    parseInt(root.querySelector("#pull-binsx").value) || 1500,
    parseInt(root.querySelector("#pull-binsy").value) || 1500,
  ];

  const readBins1d = () =>
    parseInt(root.querySelector("#pull-bins1d").value) || 1000;

  const setHeading = (bins) => {
    root.querySelector("#pull-heading").textContent =
      `Pull analysis vs GEANT4 (${bins[0]} × ${bins[1]})`;
  };
  setHeading(readBins());  // reflect the current bin inputs before any run

  for (const f of (await api.g4Files()).files) {
    const o = document.createElement("option");
    o.value = f.name; o.textContent = f.name;
    pullFileSel.appendChild(o);
  }
  if (pullFileSel.options.length) pullFileSel.value = pullFileSel.options[pullFileSel.options.length - 1].value;

  async function runPull() {
    const guard = processSelectionError();
    if (guard) { setStatus(guard, "error"); return; }
    const bins = readBins();
    setStatus(`running ${bins[0]}x${bins[1]} pull analysis (slow)...`);
    const payload = {
      g4_file: pullFileSel.value,
      E0_eV:   parseFloat(root.querySelector("#pl-e0").value) * 1e6,
      n_events: parseInt(root.querySelector("#pull-ne").value),
      n_steps:  parseInt(root.querySelector("#pull-ns").value),
      n_2d_bins: bins,
      pull_1d_n_bins: readBins1d(),
      pull_min_counts_per_bin: parseInt(root.querySelector("#pull-min").value),
      pull_1d_hist_range: [-5, 5],
      pull_fit_range:     [-30, 30],
      processes: _processesFromGlobals(),
    };

    let stopPull = false;
    const t0p = Date.now();
    (async () => {
      while (!stopPull) {
        try {
          const p = await api.simProgress();
          const elapsed = ((Date.now() - t0p) / 1000).toFixed(1);
          if (p.phase === "simulating" && p.total > 0) {
            const pct = (100 * p.step / p.total).toFixed(1);
            setStatus(`pull: simulating ${pct}% (${p.step}/${p.total}) · ${elapsed}s`);
          } else if (p.phase === "building stepping manager") {
            setStatus(`pull: building stepping manager… (${elapsed}s)`);
          } else if (p.phase === "wrapping pre-built manager") {
            setStatus(`pull: preparing simulator (fast path)… (${elapsed}s)`);
          }
        } catch (e) { /* ignore */ }
        await new Promise(r => setTimeout(r, 500));
      }
    })();

    try {
      const r = await api.pull(payload);
      renderPullResult(r);
      // Stash the sim-defining fields so Re-bin can reproduce the same
      // server-side cache keys without re-simulating.
      lastRunSimParams = {
        g4_file:  pullFileSel.value,
        E0_eV:    payload.E0_eV,
        n_events: payload.n_events,
        n_steps:  payload.n_steps,
        processes: payload.processes,
      };
      root.querySelector("#pull-rebin").disabled = false;
      setHeading(r.n_2d_bins || readBins());
      setStatus(`pull: µ=${r.mu.toFixed(3)}, σ=${r.sigma.toFixed(3)}`);
    } catch (e) {
      setStatus(`pull failed: ${e.message}`, "error");
    } finally {
      stopPull = true;
    }
  }

  // Shared post-success rendering for both a full run and a Re-bin so the two
  // paths can never visually diverge.
  function renderPullResult(r) {
    // Compute a single shared z-range across both energy panels so the
    // system and G4 heatmaps are directly comparable (otherwise each panel
    // would auto-scale independently and identical colors wouldn't mean
    // identical deposits).
    const logs = [];
    if (r.hist_sim) logs.push(_logThumb(r.hist_sim.thumbnail));
    if (r.hist_g4)  logs.push(_logThumb(r.hist_g4.thumbnail));
    let zmin = Infinity, zmax = -Infinity;
    for (const z of logs) {
      const e = _zExtent(z);
      if (e.zmin < zmin) zmin = e.zmin;
      if (e.zmax > zmax) zmax = e.zmax;
    }
    if (!isFinite(zmin) || !isFinite(zmax) || zmin === zmax) { zmin = 0; zmax = 1; }
    const { tickvals, ticktext } = _decadeTicks(zmin, zmax);

    if (r.hist_sim) {
      root.querySelector("#pull-sim-title").textContent = r.hist_sim.title || "System";
      drawEnergyMap("pull-sim-map", r.hist_sim, zmin, zmax, tickvals, ticktext);
    }
    if (r.hist_g4) drawEnergyMap("pull-g4-map", r.hist_g4, zmin, zmax, tickvals, ticktext);
    drawPullMap(r.pull_map);
    drawPullHist(r.pull_hist, r.mu, r.mu_err, r.sigma, r.sigma_err);
    root.querySelector("#pull-meta").innerHTML =
      `<span>µ₀</span><span>${r.mu.toFixed(4)} ± ${r.mu_err.toFixed(4)}</span>` +
      `<span>σ₀</span><span>${r.sigma.toFixed(4)} ± ${r.sigma_err.toFixed(4)}</span>` +
      `<span>χ²/ndof</span><span>${r.chi2_ndof.toFixed(3)}</span>` +
      `<span>bins used</span><span>${r.n_bins_used}</span>`;
  }

  async function rebinOnly() {
    if (!lastRunSimParams) {
      setStatus("run the full pull analysis first", "error");
      return;
    }
    const bins = readBins();
    setStatus(`re-binning at ${bins[0]}×${bins[1]} (no re-simulation)...`);
    const payload = {
      ...lastRunSimParams,
      n_2d_bins: bins,
      pull_1d_n_bins: readBins1d(),
      pull_min_counts_per_bin: parseInt(root.querySelector("#pull-min").value),
      pull_1d_hist_range: [-5, 5],
      pull_fit_range:     [-30, 30],
    };
    try {
      const r = await api.rebin(payload);
      renderPullResult(r);
      setHeading(r.n_2d_bins || bins);
      setStatus(`re-binned ${bins[0]}×${bins[1]}: µ=${r.mu.toFixed(3)}, σ=${r.sigma.toFixed(3)}`);
    } catch (e) {
      setStatus(`re-bin failed: ${e.message}`, "error");
    }
  }

  function _heatmapAxisArrays(thumb, x_range, y_range) {
    const ny = thumb.length;
    const nx = thumb[0] ? thumb[0].length : 0;
    const x = Array.from({length: nx}, (_, i) =>
      x_range[0] + (i + 0.5) * (x_range[1] - x_range[0]) / nx);
    const y = Array.from({length: ny}, (_, i) =>
      y_range[0] + (i + 0.5) * (y_range[1] - y_range[0]) / ny);
    return { x, y };
  }

  function _logThumb(thumb) {
    return thumb.map(row => row.map(v => (v > 0 ? Math.log10(v) : null)));
  }

  // The three heatmaps share the server's x/y range (the G4 hull), so they
  // are left to autorange and stay comparable panel to panel. The y axis is
  // deliberately NOT aspect-locked to x (scaleanchor): with a ~38 mm beam
  // axis on a wide panel that lock stretched a ±0.7 mm transverse range to
  // ±9 mm and snapped every box zoom back to the same ratio.

  function _zExtent(z) {
    let zmin = Infinity, zmax = -Infinity;
    for (const row of z) for (const v of row) {
      if (v == null) continue;
      if (v < zmin) zmin = v;
      if (v > zmax) zmax = v;
    }
    return { zmin, zmax };
  }

  function _decadeTicks(zmin, zmax) {
    const sup = "⁰¹²³⁴⁵⁶⁷⁸⁹";
    const supStr = (n) => [...String(n)].map(c => c === "-" ? "⁻" : sup[+c]).join("");
    const tickvals = [], ticktext = [];
    for (let t = Math.floor(zmin); t <= Math.ceil(zmax); t++) {
      tickvals.push(t); ticktext.push(`10${supStr(t)}`);
    }
    return { tickvals, ticktext };
  }

  function drawEnergyMap(divId, payload, zmin, zmax, tickvals, ticktext) {
    const { x, y } = _heatmapAxisArrays(payload.thumbnail, payload.x_range, payload.y_range);
    const z_log = _logThumb(payload.thumbnail);
    Plotly.react(divId, [{
      type: "heatmap", z: z_log, x, y,
      colorscale: "Jet",
      colorbar: { title: "Energy Deposition [MeV]", tickvals, ticktext },
      zmin, zmax,
      connectgaps: false, hoverongaps: false,
      hovertemplate: "x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>E=10^%{z:.2f} MeV<extra></extra>",
    }], {
      paper_bgcolor: "#161b22", plot_bgcolor: "#0e1116",
      font: { color: "#e6edf3" }, margin: { t: 20, r: 30, b: 50, l: 60 },
      xaxis: { title: "x (beam axis) [mm]", gridcolor: "#30363d", zeroline: false },
      yaxis: { title: "y (transverse axis) [mm]", gridcolor: "#30363d", zeroline: false },
    }, { responsive: true });
  }

  function drawPullMap(pm) {
    const { x, y } = _heatmapAxisArrays(pm.thumbnail, pm.x_range, pm.y_range);
    // Empty / below-threshold cells arrive as `null` from the server; Plotly
    // renders them transparent so the panel background shows through instead
    // of painting them green (the zero of the Jet scale).
    Plotly.react("pull-map", [{
      type: "heatmap", z: pm.thumbnail, x, y,
      colorscale: "Jet", zmin: -4, zmax: 4,
      connectgaps: false,
      colorbar: { title: "Pull (σ)" },
      hoverongaps: false,
      hovertemplate: "x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>pull=%{z:.2f}σ<extra></extra>",
    }], {
      paper_bgcolor: "#161b22", plot_bgcolor: "#0e1116",
      font: { color: "#e6edf3" }, margin: { t: 20, r: 30, b: 50, l: 60 },
      xaxis: { title: "x (beam axis) [mm]", gridcolor: "#30363d", zeroline: false },
      yaxis: { title: "y (transverse axis) [mm]", gridcolor: "#30363d", zeroline: false },
    }, { responsive: true });
  }

  function drawPullHist(h, mu, mu_err, sigma, sigma_err) {
    const annotation =
      `µ₀: ${mu.toFixed(3)} ± ${mu_err.toFixed(3)}<br>` +
      `σ₀: ${sigma.toFixed(3)} ± ${sigma_err.toFixed(3)}`;
    const traces = [{
      type: "bar", x: h.centers, y: h.counts, name: "Data",
      marker: { color: "#3a3f47" }, hoverinfo: "skip",
    }];
    if (h.fit_x && h.fit_y) {
      traces.push({
        type: "scatter", mode: "lines", x: h.fit_x, y: h.fit_y, name: "Fit",
        line: { color: "#ff7b00", width: 2.5 }, hoverinfo: "skip",
      });
    }
    Plotly.react("pull-hist", traces, {
      paper_bgcolor: "#161b22", plot_bgcolor: "#0e1116",
      font: { color: "#e6edf3" }, margin: { t: 20, r: 20, b: 50, l: 60 },
      xaxis: { title: "Pull Values (Nσ)", gridcolor: "#30363d", range: [-5, 5] },
      yaxis: { title: "Bins in y-x plane", gridcolor: "#30363d" },
      bargap: 0,
      legend: { x: 0.98, y: 0.98, xanchor: "right", yanchor: "top",
                bgcolor: "rgba(22,27,34,0.6)" },
      annotations: [{
        text: annotation, xref: "paper", yref: "paper",
        x: 0.02, y: 0.98, xanchor: "left", yanchor: "top",
        showarrow: false, align: "left",
        font: { family: "ui-monospace,monospace", size: 13 },
      }],
    }, { responsive: true });
  }

  root.querySelector("#pull-run").addEventListener("click", runPull);
  root.querySelector("#pull-rebin").addEventListener("click", rebinOnly);
}
