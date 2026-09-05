// Unified generators tab — overlays MC, NN, and analytical PDF.
// G4 truth is added only when the global selector mode is 'bin' (otherwise the
// cell cut is undefined). In bin mode, MC and NN are conditioned per-event on
// the actual G4 rows inside the cell (server-side, via the same cut that
// builds the G4 histogram) and draw one sample per in-bin event; the
// analytical PDF stays at the pinned point. The NN choice comes from the
// global per-stage generator dropdown; if the user picks "MC baseline" for
// the stage, the NN trace is skipped and only MC + PDF (+ G4) are drawn.
//
// MC and NN are binned client-side on shared edges built from the union of
// their sample ranges, so the two histograms align exactly bin-for-bin. The
// residual panel below the main overlay plots (sample_density − pdf) / pdf
// at each bin centre for both MC and NN, sharing the same x-axis range.
import { api, setStatus } from "../api.js";
import {
  getPhaseSpacePoint, getStageGenerator,
  getSelectorMode, getHalfwidthDex, onChange,
} from "../state.js";

const STAGE_LABELS = {
  length:      "Step length",
  continuous:  "Continuous loss",
  secondary:   "Secondary energy",
};

const STAGE_TO_G4COL = {
  length:      "StepLength",
  continuous:  "ContinuousLoss",
  secondary:   "SecondaryLoss",
};

const STAGES_WITH_L = new Set(["continuous"]);

export async function mountGenerators(root) {
  root.innerHTML = `
    <div class="card">
      <h2>Sampling-algorithm overlays</h2>
      <div class="controls">
        <label>stage</label>
        <select id="gen-stage">
          ${Object.entries(STAGE_LABELS)
            .map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}
        </select>
        <label>n samples</label>
        <input id="gen-n" type="number" value="100000" min="500" max="1000000" step="1000">
        <label>bins</label>
        <input id="gen-bins" type="number" value="120" min="20" max="400">
        <label>y-mode</label>
        <select id="gen-ymode">
          <option value="density" selected>density</option>
          <option value="counts">counts</option>
        </select>
        <label>G4 file</label>
        <select id="gen-file"></select>
        <button id="gen-resample" type="button" style="margin-left:8px"
                title="Re-draw fresh MC/NN samples without changing any input.">↻ Resample</button>
        <span id="gen-active-nn" class="muted" style="margin-left:8px"></span>
      </div>
      <div id="gen-plot" class="plot plot-tall"></div>
      <div id="gen-resid" class="plot" style="height:240px;margin-top:6px"></div>
      <div id="gen-meta" class="kv"></div>
    </div>
  `;

  const fileSel = root.querySelector("#gen-file");
  for (const f of (await api.g4Files()).files) {
    const o = document.createElement("option");
    o.value = f.name; o.textContent = f.name;
    fileSel.appendChild(o);
  }
  if (fileSel.options.length) fileSel.value = fileSel.options[fileSel.options.length - 1].value;

  // Latest run snapshot — used by the residual panel and re-render on
  // density/counts toggle without re-sampling.
  let lastDraw = null;
  // Token guards against out-of-order async results when the user clicks rapidly.
  let runToken = 0;

  function _updateNnTag() {
    const stage = root.querySelector("#gen-stage").value;
    const choice = getStageGenerator(stage);
    const tag = root.querySelector("#gen-active-nn");
    if (choice.kind === "nn") {
      tag.textContent = `(NN: ${choice.class_name} · ${choice.run_id})`;
      tag.className = "good";
    } else {
      tag.textContent = "(MC baseline)";
      tag.className = "muted";
    }
  }

  // Bin a flat array of positive samples on log-spaced shared edges.
  function _histOn(samples, edges) {
    const counts = new Array(edges.length - 1).fill(0);
    if (!samples?.length) return counts;
    // Log10 binary search on edges.
    const logEdges = edges.map(e => Math.log10(e));
    for (const v of samples) {
      if (!(v > 0) || !isFinite(v)) continue;
      const lv = Math.log10(v);
      // Manual binary search.
      let lo = 0, hi = logEdges.length - 1;
      if (lv < logEdges[0] || lv > logEdges[hi]) continue;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (lv >= logEdges[mid]) lo = mid; else hi = mid;
      }
      counts[lo]++;
    }
    return counts;
  }

  function _bandFromSamples(arrays) {
    let min = Infinity, max = -Infinity;
    for (const a of arrays) {
      if (!a) continue;
      for (const v of a) {
        if (!(v > 0) || !isFinite(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min) || !isFinite(max) || min >= max) return null;
    return [min, max];
  }

  function _sharedLogEdges(min, max, nbins) {
    const lo = Math.log10(min), hi = Math.log10(max);
    const out = new Array(nbins + 1);
    for (let i = 0; i <= nbins; i++) out[i] = Math.pow(10, lo + (hi - lo) * i / nbins);
    return out;
  }

  async function run() {
    const ps = getPhaseSpacePoint();
    const stage = root.querySelector("#gen-stage").value;
    const needsL = STAGES_WITH_L.has(stage);
    if (ps.E_eV == null || (needsL && ps.L_m == null)) {
      // Quietly wait until the user pins a point on the G4 browser tab.
      return;
    }
    const n     = root.querySelector("#gen-n").value;
    const bins  = parseInt(root.querySelector("#gen-bins").value, 10);
    const mode  = getSelectorMode();
    const hw    = getHalfwidthDex();
    const file  = fileSel.value;
    const choice = getStageGenerator(stage);

    const myToken = ++runToken;
    setStatus(`sampling ${stage}...`);
    _updateNnTag();

    // Bin-conditioning params shared by the G4 slice, the MC sampler, and
    // the NN generator. When present, the server conditions each MC/NN
    // draw on one actual in-bin G4 event (per-row E, L) instead of
    // repeating the pinned point, and draws exactly one sample per in-bin
    // event so the counts histograms are directly comparable. The L cut
    // applies only to stages conditioned on L — for length/secondary it
    // would degenerate the histogram into the pinned L bin itself.
    let binParams = null;
    if (mode === "bin") {
      binParams = { file, e_center: ps.E_eV, e_halfwidth: hw };
      if (needsL) {
        binParams.l_center = ps.L_m;
        binParams.l_halfwidth = hw;
      }
    }

    // MC always. L is only meaningful for the stages conditioned on it.
    const mcParams = { stage, E: ps.E_eV, n, bins, log_x: 1,
                       ...(binParams ?? {}) };
    if (needsL) mcParams.L = ps.L_m;
    const mcPromise = api.sample(mcParams);

    // NN only if the global choice asks for it.
    let nnPromise = Promise.resolve(null);
    if (choice.kind === "nn") {
      const genParams = {
        stage, class_name: choice.class_name, checkpoint: choice.checkpoint,
        E: ps.E_eV, n, bins, log_x: 1, ...(binParams ?? {}),
      };
      if (needsL) genParams.L = ps.L_m;
      nnPromise = api.generate(genParams)
        .catch(e => ({ error: e.message, edges: [], counts: [], samples: [] }));
    }

    // G4 truth histogram, only in bin mode (same cell cut as binParams).
    let g4Promise = Promise.resolve(null);
    if (mode === "bin") {
      const g4Params = { ...binParams, col: STAGE_TO_G4COL[stage], bins, log_x: 1 };
      g4Promise = api.g4Hist1d(g4Params);
    }

    // The NN promise already degrades to an error object; the MC and G4
    // fetches don't, and run() is fired-and-forgotten from event handlers —
    // an unhandled rejection here would leave "sampling ..." up forever.
    let mc, nn, g4;
    try {
      [mc, nn, g4] = await Promise.all([mcPromise, nnPromise, g4Promise]);
    } catch (err) {
      if (myToken === runToken) setStatus(`${stage} failed: ${err.message}`);
      return;
    }
    if (myToken !== runToken) return;  // a newer run has since started.

    // Shared edges from the union of MC and NN sample ranges (positive only).
    const band = _bandFromSamples([mc.samples, nn?.samples]);
    let edges = null;
    if (band) edges = _sharedLogEdges(band[0], band[1], bins);

    // Analytical PDF curve — uses MC/NN union range to choose x bounds.
    let x_min = 1e-12, x_max = 1e6;
    if (band) {
      x_min = Math.max(band[0] / 5, 1e-30);
      x_max = band[1] * 5;
    } else if (mc.edges?.length) {
      x_min = Math.max(mc.edges[0] / 5, 1e-30);
      x_max = mc.edges[mc.edges.length - 1] * 5;
    }
    // Sample the analytical PDF densely enough that consecutive points are
    // within ~1 visible pixel even on a wide log range. A fixed point count
    // spread thinly over a wide range leaves the curve looking jagged
    // (saw-tooth in log-y) at typical view spans of 1-2 decades. 400
    // points/decade keeps the curve visually smooth without ballooning the
    // JSON payload.
    const _decadesSpan = Math.max(1, Math.log10(x_max) - Math.log10(x_min));
    const pdfNPoints = Math.min(20000, Math.max(2000, Math.round(400 * _decadesSpan)));
    const pdfParams = { stage, E: ps.E_eV, x_min, x_max, n_points: pdfNPoints,
                        log_x: 1 };
    if (ps.L_m != null) pdfParams.L = ps.L_m;
    const pdf = await api.pdfCurve(pdfParams)
      .catch(err => ({ error: err.message, x: [], pdf: [] }));
    if (myToken !== runToken) return;

    lastDraw = { stage, g4, mc, nn, pdf, mode, choice, edges, band };
    draw(lastDraw);

    const parts = [
      `MC=${mc.samples_n ?? 0}`,
      nn ? `NN=${nn.samples_n ?? 0}` : null,
      g4 ? `G4=${g4.n}` : null,
      `regime=${pdf.regime ?? "—"}`,
    ].filter(Boolean);
    setStatus(`${stage}: ${parts.join(", ")}`);
  }

  function _binCentersAndWidths(edges) {
    const centers = new Array(edges.length - 1);
    const widths  = new Array(edges.length - 1);
    for (let i = 0; i < centers.length; i++) {
      centers[i] = 0.5 * (edges[i] + edges[i + 1]);
      widths[i]  = edges[i + 1] - edges[i];
    }
    return { centers, widths };
  }

  // Linear interpolation on a sorted x-grid in log10 space (the analytical
  // PDF curve is log-x sampled; interpolating in log space avoids visible
  // kinks at the bin centres for sub-decade bin spacing).
  function _interpLogX(xs, ys, x) {
    if (!xs?.length) return NaN;
    if (x <= xs[0]) return ys[0];
    const last = xs.length - 1;
    if (x >= xs[last]) return ys[last];
    const lx = Math.log10(x);
    let lo = 0, hi = last;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (Math.log10(xs[mid]) <= lx) lo = mid; else hi = mid;
    }
    const lx0 = Math.log10(xs[lo]), lx1 = Math.log10(xs[hi]);
    const t = (lx - lx0) / (lx1 - lx0);
    return ys[lo] + t * (ys[hi] - ys[lo]);
  }

  // Trapezoidal cumulative integral of `pdf` over `xs`, normalized to 1.
  // Returns null if the support is empty or the integral is non-positive.
  // The analytical PDF is already conditioned on positive loss (see the
  // continuous branch in playground.services.sampling), so the resulting
  // CDF is the conditional CDF that aligns with the MC histogram, which
  // also discards zero-loss samples.
  function _cdfFromPdf(xs, pdf) {
    if (!xs?.length || xs.length !== pdf.length) return null;
    const N = xs.length;
    const cdf = new Array(N).fill(0);
    for (let i = 1; i < N; i++) {
      const a = pdf[i - 1], b = pdf[i];
      const ya = (isFinite(a) && a > 0) ? a : 0;
      const yb = (isFinite(b) && b > 0) ? b : 0;
      cdf[i] = cdf[i - 1] + 0.5 * (ya + yb) * (xs[i] - xs[i - 1]);
    }
    const tot = cdf[N - 1];
    if (!(tot > 0) || !isFinite(tot)) return null;
    for (let i = 0; i < N; i++) cdf[i] /= tot;
    return cdf;
  }

  // One-sample two-sided KS distance + asymptotic p-value between `samples`
  // and the analytical CDF (xs, cdf). Empirical CDF jumps from i/n to (i+1)/n
  // at each sorted sample, so D = max over i of max(|F(x_i)−i/n|, |F(x_i)−(i+1)/n|).
  // p-value uses the standard Kolmogorov series 2·Σ (−1)^(k−1) exp(−2 k² λ²)
  // with λ = √n · D (1-sample asymptotic form).
  function _ksOneSampleVsCdf(samples, xs, cdf) {
    if (!samples?.length || !xs?.length || !cdf?.length) return null;
    const xLo = xs[0], xHi = xs[xs.length - 1];
    const pos = [];
    for (const v of samples) {
      if (!(v > 0) || !isFinite(v)) continue;
      if (v < xLo || v > xHi) continue;
      pos.push(v);
    }
    if (pos.length < 2) return null;
    pos.sort((a, b) => a - b);
    const n = pos.length;
    let D = 0;
    for (let i = 0; i < n; i++) {
      const F = _interpLogX(xs, cdf, pos[i]);
      const lo = i / n, hi = (i + 1) / n;
      const d1 = Math.abs(F - lo), d2 = Math.abs(F - hi);
      if (d1 > D) D = d1;
      if (d2 > D) D = d2;
    }
    const lam = Math.sqrt(n) * D;
    let p;
    if (lam < 1e-6) {
      p = 1;
    } else {
      let s = 0;
      for (let k = 1; k < 200; k++) {
        const term = 2 * ((k % 2 === 1) ? 1 : -1) * Math.exp(-2 * k * k * lam * lam);
        s += term;
        if (Math.abs(term) < 1e-15) break;
      }
      p = Math.max(0, Math.min(1, s));
    }
    return { D, p, n };
  }

  // Compact integer formatter: 1, 10, 100, 1k, 10k, 100k, 1M, ...
  // Used in legend labels so n=100000 reads as "100k", not "100000".
  function _fmtCount(n) {
    if (n == null || !isFinite(n)) return "—";
    n = Math.round(n);
    if (n < 1000) return String(n);
    const units = [
      { v: 1e9, s: "B" },
      { v: 1e6, s: "M" },
      { v: 1e3, s: "k" },
    ];
    for (const { v, s } of units) {
      if (Math.abs(n) >= v) {
        const x = n / v;
        // Drop trailing zeros: 1.0k → 1k, 1.5k → 1.5k.
        const str = (Math.abs(x) >= 10) ? x.toFixed(0)
                  : (Math.abs(x * 10 - Math.round(x * 10)) < 1e-9)
                      ? (x.toFixed(1).replace(/\.0$/, ""))
                      : x.toFixed(1);
        return `${str}${s}`;
      }
    }
    return String(n);
  }

  function _trace(name, xs, ys, color, dash = null, width = 2) {
    return {
      type: "scatter", mode: "lines", x: xs, y: ys, name,
      line: { color, width, ...(dash ? { dash } : {}) },
    };
  }

  // Build a stair-step polyline from bin edges + per-bin heights so MC/NN
  // render as a true histogram outline (matches `histtype='step'` in
  // matplotlib). The walls are added as vertical segments anchored to the
  // bin edges instead of bin centres so the bars align with the residual
  // panel below.
  function _stepFromEdges(edges, heights) {
    const xs = [], ys = [];
    for (let i = 0; i < heights.length; i++) {
      xs.push(edges[i],   edges[i + 1]);
      ys.push(heights[i], heights[i]);
    }
    return { xs, ys };
  }
  function _stepTrace(name, edges, heights, color, width = 2) {
    const { xs, ys } = _stepFromEdges(edges, heights);
    return {
      type: "scatter", mode: "lines", x: xs, y: ys, name,
      line: { color, width, shape: "linear" },
    };
  }

  function draw(d) {
    const { stage, g4, mc, nn, pdf, mode, choice, edges, band } = d;
    const yMode = root.querySelector("#gen-ymode").value;  // 'density' | 'counts'
    const traces = [];

    let plotCenters = null, plotWidths = null;
    if (edges) {
      const cw = _binCentersAndWidths(edges);
      plotCenters = cw.centers; plotWidths = cw.widths;
    }

    // Helper: bin samples → counts / density on shared edges.
    function _binned(samples) {
      if (!edges || !samples?.length) return null;
      const counts = _histOn(samples, edges);
      const N = counts.reduce((a, b) => a + b, 0);
      if (N === 0) return { counts, density: counts.map(() => 0) };
      const density = counts.map((c, i) => c / (N * plotWidths[i]));
      return { counts, density, N };
    }

    // G4 (only in bin mode) — use its own edges since it ships pre-binned.
    if (g4 && g4.edges?.length) {
      const cw = _binCentersAndWidths(g4.edges);
      const N = g4.counts.reduce((a, b) => a + b, 0);
      const y = yMode === "counts"
        ? g4.counts
        : g4.counts.map((c, i) => N > 0 ? c / (N * cw.widths[i]) : 0);
      traces.push(_stepTrace(`G4 (n=${_fmtCount(g4.n)})`, g4.edges, y, "#58a6ff"));
    }

    const mcBins = _binned(mc.samples);
    if (mcBins) {
      const y = yMode === "counts" ? mcBins.counts : mcBins.density;
      traces.push(_stepTrace(`MC (n=${_fmtCount(mc.samples_n)})`, edges, y, "#3fb950"));
    }

    const nnBins = (nn && !nn.error) ? _binned(nn.samples) : null;
    if (nnBins) {
      const y = yMode === "counts" ? nnBins.counts : nnBins.density;
      traces.push(_stepTrace(`NN ${choice.class_name} (n=${_fmtCount(nn.samples_n)})`,
                             edges, y, "#d55e00"));
    }

    // Analytical PDF curve: density domain by construction. For counts mode
    // scale to N_mc · width(x) at each plotted x, using mc's bin widths
    // interpolated at the analytical x grid.
    let pdfX = [], pdfY = [];
    if (pdf.x?.length && !pdf.error) {
      pdfX = pdf.x;
      if (yMode === "density" || !mcBins) {
        pdfY = pdf.pdf;
      } else {
        // Approximate per-x bin width by reusing the nearest mc-bin width.
        pdfY = pdf.x.map((x, i) => {
          // Pick the closest center in log space.
          const lx = Math.log10(x);
          let lo = 0, hi = plotCenters.length - 1;
          while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (Math.log10(plotCenters[mid]) <= lx) lo = mid; else hi = mid;
          }
          const w = plotWidths[lo];
          return pdf.pdf[i] * mcBins.N * w;
        });
      }
      traces.push(_trace(`Analytical (${pdf.regime})`, pdfX, pdfY, "#d29922", "dash"));
    }

    // X-axis range: union of MC and NN sample ranges, padded by ~0.3 dex.
    let xRange = null;
    if (band) {
      const padLo = Math.log10(band[0]) - 0.3;
      const padHi = Math.log10(band[1]) + 0.3;
      xRange = [padLo, padHi];
    }

    Plotly.react("gen-plot", traces, {
      paper_bgcolor: "#161b22", plot_bgcolor: "#0e1116",
      font: { color: "#e6edf3" }, margin: { t: 20, r: 30, b: 30, l: 60 },
      xaxis: { title: stage, type: "log", gridcolor: "#30363d", exponentformat: "e",
               range: xRange },
      yaxis: { title: yMode, type: "log", gridcolor: "#30363d", exponentformat: "e" },
      legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)" },
    }, { responsive: true });

    // ─── Residual panel ─────────────────────────────────────────────
    // (sample - pdf) / pdf at each bin centre. Uses the same xRange.
    //
    // Y-axis is symlog: residuals span many decades on both signs (small
    // overshoots near the peak, huge relative errors in the tails). Plotly
    // has no native symlog axis, so the values are transformed as
    //     y_t = sign(y) · log10(1 + |y| / linthresh)
    // and plotted on a linear axis whose ticks are inverse-transformed back
    // to the residual values the reader expects (…, −10, −1, −0.1, 0, 0.1,
    // 1, 10, …). linthresh sets the half-width of the linear region around
    // zero — 1e-2 keeps the linear band tight without collapsing genuinely
    // small residuals into noise.
    const LINTHRESH = 1e-2;
    const _symlog = (y) => {
      if (y == null || !isFinite(y)) return null;
      const s = y < 0 ? -1 : 1;
      return s * Math.log10(1 + Math.abs(y) / LINTHRESH);
    };
    const residTraces = [];
    if (plotCenters && pdf.x?.length && !pdf.error) {
      const pdfAtCenters = plotCenters.map(c => _interpLogX(pdf.x, pdf.pdf, c));
      function _residOf(bins, color, name) {
        if (!bins) return;
        // Compare the same y-mode as the main plot for visual coherence:
        // density vs density, counts vs (pdf · N · width).
        const obs = yMode === "counts" ? bins.counts : bins.density;
        const ref = yMode === "counts"
          ? pdfAtCenters.map((p, i) => p * bins.N * plotWidths[i])
          : pdfAtCenters;
        const res = obs.map((v, i) => {
          const r = ref[i];
          if (!isFinite(r) || r === 0) return null;
          return (v - r) / r;
        });
        const resT = res.map(_symlog);
        const hover = res.map(r => r == null ? "" : r.toExponential(3));
        residTraces.push({
          type: "scatter", mode: "lines+markers",
          x: plotCenters, y: resT, name,
          customdata: hover,
          hovertemplate: "%{x:.3e}<br>residual: %{customdata}<extra></extra>",
          line: { color, width: 1.5 }, marker: { color, size: 4 },
          connectgaps: false,
        });
      }
      _residOf(mcBins, "#3fb950", "MC − PDF / PDF");
      _residOf(nnBins, "#d55e00", "NN − PDF / PDF");
    }

    // Symlog tick ladder: decade ticks on both signs plus 0.
    const tickResids = [-100, -10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10, 100];
    const tickvals = tickResids.map(_symlog);
    const ticktext = tickResids.map(v =>
      v === 0 ? "0" : (Math.abs(v) >= 1 ? `${v}` : `${v}`)
    );
    // Y-range: snap to the smallest decade that contains the data, with a
    // sensible default when there are no residuals yet.
    let yResMax = Math.log10(1 + 1 / LINTHRESH);  // |y|=1 transformed
    for (const tr of residTraces) {
      for (const v of tr.y) {
        if (v == null || !isFinite(v)) continue;
        if (Math.abs(v) > yResMax) yResMax = Math.abs(v);
      }
    }
    const yResRange = [-yResMax * 1.05, yResMax * 1.05];

    Plotly.react("gen-resid", residTraces, {
      paper_bgcolor: "#161b22", plot_bgcolor: "#0e1116",
      font: { color: "#e6edf3" }, margin: { t: 10, r: 30, b: 40, l: 60 },
      xaxis: { title: stage, type: "log", gridcolor: "#30363d",
               exponentformat: "e", range: xRange },
      yaxis: { title: "(sample − pdf) / pdf  [symlog]", gridcolor: "#30363d",
               zeroline: true, zerolinecolor: "#666",
               tickvals, ticktext, range: yResRange },
      legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1,
                 yref: "y", y0: 0, y1: 0,
                 line: { color: "#888", width: 1, dash: "dot" } }],
    }, { responsive: true });

    // ─── KS distance & p-value vs the analytical CDF ────────────────
    // One-sample test: MC samples (and NN samples, when loaded) vs the
    // conditional analytical CDF built by trapezoidal integration of the
    // PDF. See the disclaimer in the meta panel below for the caveats.
    let ksMc = null, ksNn = null;
    if (pdf.x?.length && !pdf.error) {
      const cdf = _cdfFromPdf(pdf.x, pdf.pdf);
      if (cdf) {
        if (mc.samples?.length) ksMc = _ksOneSampleVsCdf(mc.samples, pdf.x, cdf);
        if (nn && !nn.error && nn.samples?.length) {
          ksNn = _ksOneSampleVsCdf(nn.samples, pdf.x, cdf);
        }
      }
    }
    const _fmtKs = (r) => r == null ? "—" :
      `D = ${r.D.toExponential(3)} · p = ${r.p.toExponential(3)} · n = ${_fmtCount(r.n)}`;

    const meta = root.querySelector("#gen-meta");
    const ps = getPhaseSpacePoint();
    meta.innerHTML =
      `<span>mode</span><span>${mode}${mode === "bin" ? ` · ±${getHalfwidthDex()} dex` : ""}</span>` +
      `<span>conditioning</span><span>${mc.conditioning ?? "point"}${mc.conditioning === "bin" ? ` · ${_fmtCount(mc.bin_n)} G4 events` : ""}</span>` +
      `<span>E</span><span>${ps.E_eV?.toExponential(3) ?? "—"} eV</span>` +
      `<span>L</span><span>${ps.L_m?.toExponential(3) ?? "—"} m</span>` +
      `<span>NN</span><span>${choice.kind === "nn" ? `${choice.class_name} · ${choice.run_id}` : "—"}</span>` +
      `<span>regime</span><span>${pdf.regime ?? "—"}</span>` +
      `<span>KS MC vs Analytical</span><span>${_fmtKs(ksMc)}</span>` +
      `<span>KS NN vs Analytical</span><span>${_fmtKs(ksNn)}</span>` +
      `<span class="muted" style="grid-column:1/-1;font-size:0.85em;line-height:1.4;margin-top:4px">` +
        `<b>Disclaimer.</b> The KS distance is one-sample (samples vs the analytical CDF) ` +
        `with the analytical CDF built by trapezoidal integration of the displayed log-spaced ` +
        `PDF and conditioned on positive support (the zero-loss delta is dropped to match the ` +
        `MC histogram, which discards v≤0). The p-value uses the asymptotic Kolmogorov series ` +
        `<i>p = 2 Σ (−1)<sup>k−1</sup> exp(−2 k² λ²)</i>, λ = √n · D — this assumes i.i.d. ` +
        `samples drawn from the reference CDF. At large n (here typically 10<sup>5</sup>) the ` +
        `test becomes hypersensitive: tiny numerical artefacts in the analytical PDF (Glandz ` +
        `scaling, finite-grid convolution, the dropped delta) drive D well above the rejection ` +
        `threshold even when the shapes overlay correctly. Read D as a quantitative shape ` +
        `mismatch rather than a hypothesis test; the p-value is informational only.` +
      `</span>` +
      (nn?.error ? `<span class="warn">NN error</span><span class="warn">${nn.error}</span>` : ``) +
      (pdf.error ? `<span class="warn">PDF error</span><span class="warn">${pdf.error}</span>` : ``);
  }

  root.querySelector("#gen-stage").addEventListener("change", () => {
    _updateNnTag();
    run();
  });
  // y-mode and bins/n change → re-render existing samples without re-sampling
  // when possible (y-mode is purely cosmetic).
  root.querySelector("#gen-ymode").addEventListener("change", () => {
    if (lastDraw) draw(lastDraw); else run();
  });
  root.querySelector("#gen-bins").addEventListener("change", run);
  root.querySelector("#gen-n").addEventListener("change", run);
  fileSel.addEventListener("change", run);
  // Force a fresh draw with the current inputs unchanged.
  root.querySelector("#gen-resample").addEventListener("click", run);

  // Auto-run whenever phase-space, selector mode, or generator selection
  // changes (debounced so a flurry of state updates doesn't re-fire).
  let debounceTimer = null;
  onChange(() => {
    _updateNnTag();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(run, 60);
  });
  _updateNnTag();
}
