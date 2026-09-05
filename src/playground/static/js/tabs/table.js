// Table tab — shows a resulting table of the most recent simulation run
// under the Trajectory / Pull / Pairwise tabs. A "table" selector switches
// between two sources:
//   - System: the simulator's merged states+step-features DataFrame
//     (TrackSimulator.states_and_steps_features_df), kept on the last-run
//     simulator;
//   - GEANT4: the truth tracks the run was compared against (only available
//     after a Pull / Pairwise run). The two have different column schemas, so
//     switching source rebuilds the column chooser.
//
// The server keeps the full DataFrames; this tab fetches a paginated slice
// from /api/table/last?which=sys|g4 and renders it as an HTML table.
// "Refresh" re-fetches so a sim run after this tab was first opened shows up;
// the page size + prev/next controls page through the rows.
//
// Two filters layer on top:
//   - query: a pandas df.query() expression applied server-side (it spans all
//     rows, which the client doesn't hold), e.g.
//     "KineticEnergy > 5e7 and StepLength < 1e-5". Re-fetches on Apply.
//   - column chooser: client-side checkboxes that hide/show columns instantly
//     without a refetch (the full row payload is already capped at ≤5000).
import { api, setStatus } from "../api.js";
import { formatEng } from "../state.js";

const PAGE_SIZES = [100, 200, 500, 1000, 5000];

function _fmtCell(v) {
  if (v === null || v === undefined) return "";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return String(v);
  // Engineering notation keeps energies/lengths readable across many decades.
  return formatEng(v);
}

export async function mountTable(root) {
  root.innerHTML = `
    <div class="card">
      <h2>Resulting table</h2>
      <p class="muted">
        The tables of the most recent simulation run under the Trajectory,
        Pull, or Pairwise tab. <b>System</b> is the simulator's merged states +
        step-features table (one row per step: event/step indices, lab-frame
        X/Y/Z, kinetic energy, and all step features). <b>GEANT4</b> is the
        truth tracks the run was compared against — available only after a Pull
        or Pairwise comparison, and with a different column schema.
      </p>
      <div class="controls">
        <button id="tbl-refresh" class="primary">Refresh</button>
        <label>table</label>
        <select id="tbl-which">
          <option value="sys" selected>System</option>
          <option value="g4">GEANT4</option>
        </select>
        <label>rows per page</label>
        <select id="tbl-page-size">
          ${PAGE_SIZES.map(n => `<option value="${n}"${n === 200 ? " selected" : ""}>${n}</option>`).join("")}
        </select>
        <button id="tbl-prev">◀ Prev</button>
        <span id="tbl-range" class="muted"></span>
        <button id="tbl-next">Next ▶</button>
      </div>
      <div class="controls">
        <label>query</label>
        <input id="tbl-query" type="text" style="width:380px"
               placeholder="e.g. KineticEnergy > 5e7 and StepLength < 1e-5">
        <button id="tbl-apply">Apply</button>
        <button id="tbl-clear">Clear</button>
      </div>
      <div id="tbl-cols" class="col-chooser"></div>
      <div id="tbl-info" class="muted"></div>
    </div>

    <div class="card">
      <div id="tbl-wrap" style="overflow:auto; max-height:70vh"></div>
    </div>

    <div class="card">
      <h2>Notebook</h2>
      <p class="muted">
        Run Python against the tables. <code>df_sys</code> is the System table
        and <code>df_g4</code> is the GEANT4 table (a pandas DataFrame copy each;
        <code>df_g4</code> is <code>None</code> if the last run had no G4
        reference). <code>df</code> aliases <code>df_sys</code>. <code>pd</code>,
        <code>np</code>, <code>plt</code> and <code>ts</code> (the
        TrackSimulator) are in scope. Variables persist across cells; matplotlib
        figures render inline. <b>Shift+Enter</b> runs a cell (and moves to the
        next), <b>Ctrl/⌘+Enter</b> runs in place. Executes locally on the
        server. The tables are captured when the kernel starts — after running a
        new simulation, hit <i>Reset kernel</i> to rebind them (this also clears
        your variables).
      </p>
      <div class="controls">
        <button id="nb-add">+ Cell</button>
        <button id="nb-run-all">Run all</button>
        <button id="nb-reset">Reset kernel</button>
        <span id="nb-status" class="muted"></span>
      </div>
      <div id="nb-cells"></div>
    </div>
  `;

  const refreshBtn = root.querySelector("#tbl-refresh");
  const whichSel = root.querySelector("#tbl-which");
  const pageSizeSel = root.querySelector("#tbl-page-size");
  const prevBtn = root.querySelector("#tbl-prev");
  const nextBtn = root.querySelector("#tbl-next");
  const rangeEl = root.querySelector("#tbl-range");
  const queryInput = root.querySelector("#tbl-query");
  const applyBtn = root.querySelector("#tbl-apply");
  const clearBtn = root.querySelector("#tbl-clear");
  const colsHost = root.querySelector("#tbl-cols");
  const info = root.querySelector("#tbl-info");
  const wrap = root.querySelector("#tbl-wrap");

  let offset = 0;
  let total = 0;
  let curColumns = [];        // full column set from the last response
  let curRows = [];           // rows of the current page (all columns)
  let visibleCols = null;     // Set of shown column names; null until first load
  let colsSource = null;      // which schema ('sys'/'g4') the chooser was built for

  // ── column chooser (client-side) ────────────────────────────────────────
  function buildColChooser(columns) {
    if (visibleCols === null) visibleCols = new Set(columns);
    colsHost.innerHTML =
      `<span class="muted">columns:</span> ` +
      `<button id="tbl-cols-all" class="mini">all</button> ` +
      `<button id="tbl-cols-none" class="mini">none</button> ` +
      columns.map(c =>
        `<label class="col-check"><input type="checkbox" data-col="${c}"` +
        `${visibleCols.has(c) ? " checked" : ""}> ${c}</label>`
      ).join("");

    colsHost.querySelectorAll("input[type=checkbox]").forEach(cb => {
      cb.addEventListener("change", () => {
        if (cb.checked) visibleCols.add(cb.dataset.col);
        else visibleCols.delete(cb.dataset.col);
        rerender();
      });
    });
    colsHost.querySelector("#tbl-cols-all").addEventListener("click", () => {
      visibleCols = new Set(columns); buildColChooser(columns); rerender();
    });
    colsHost.querySelector("#tbl-cols-none").addEventListener("click", () => {
      visibleCols = new Set(); buildColChooser(columns); rerender();
    });
  }

  // With `width:100%`, fewer columns stretch each one wide while a fixed font
  // looks lost in the whitespace. Scale the font up as columns are hidden:
  // full schema → 12px, ramping to 22px when only a couple are shown.
  function _fontForColumns(nVisible) {
    const nTotal = curColumns.length || nVisible;
    const px = 12 + (nTotal - nVisible) * 1.0;
    return Math.round(Math.min(22, Math.max(12, px)));
  }

  function _renderTable(columns, rows) {
    if (!columns.length) {
      wrap.innerHTML = `<p class="muted">no columns selected</p>`;
      return;
    }
    if (!rows.length) {
      wrap.innerHTML = `<p class="muted">no rows</p>`;
      return;
    }
    const head = `<tr>${columns.map(c => `<th>${c}</th>`).join("")}</tr>`;
    const body = rows.map(r =>
      `<tr>${r.map(v => `<td>${_fmtCell(v)}</td>`).join("")}</tr>`
    ).join("");
    const fontPx = _fontForColumns(columns.length);
    wrap.innerHTML =
      `<table class="data-table" style="font-size:${fontPx}px">` +
      `<thead>${head}</thead><tbody>${body}</tbody></table>`;
  }

  // Re-render the current page through the visible-column filter (no refetch).
  function rerender() {
    const idx = curColumns
      .map((c, i) => (visibleCols && visibleCols.has(c) ? i : -1))
      .filter(i => i >= 0);
    const cols = idx.map(i => curColumns[i]);
    const rows = curRows.map(r => idx.map(i => r[i]));
    _renderTable(cols, rows);
  }

  async function load() {
    const limit = parseInt(pageSizeSel.value);
    const query = queryInput.value.trim();
    const which = whichSel.value;
    info.innerHTML = `<span class="warn">loading…</span>`;
    try {
      const r = await api.tableLast({ offset, limit, query, which });
      if (!r.available) {
        wrap.innerHTML = "";
        rangeEl.textContent = "";
        colsHost.innerHTML = "";
        info.innerHTML = `<span class="warn">no simulation has been run yet — run one in the Trajectory, Pull, or Pairwise tab, then click Refresh.</span>`;
        setStatus("table: no simulation yet");
        return;
      }

      // Reflect G4 availability in the selector so the user can't pick a
      // table that doesn't exist for this run.
      const g4Opt = whichSel.querySelector('option[value="g4"]');
      if (g4Opt) g4Opt.disabled = !r.g4_available;

      // The System and GEANT4 tables have different schemas, so rebuild the
      // column chooser whenever the active source changes (or on first load).
      curColumns = r.columns;
      if (colsSource !== r.which) {
        visibleCols = null;            // show all columns of the new schema
        buildColChooser(curColumns);
        colsSource = r.which;
      }

      const src = r.source ? `source: ${r.source}` : "source: —";
      const tbl = r.which === "g4"
        ? `table: GEANT4 (${r.g4_file || "?"})`
        : "table: System";
      const head = `${tbl} · ${src} · E0=${formatEng(r.E0_eV)} eV · n_events=${r.n_events} · n_steps=${r.n_steps}`;

      if (r.query_error) {
        curRows = [];
        rerender();
        rangeEl.textContent = "";
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        info.innerHTML = `<span class="error">query error — ${r.query_error}</span>`;
        setStatus(`table: query error`, "error");
        return;
      }

      total = r.total_rows;
      offset = r.offset;   // server clamps; trust it
      curRows = r.rows;
      const shown = r.rows.length;
      const from = total ? offset + 1 : 0;
      const to = offset + shown;
      rangeEl.textContent = `${from}–${to} of ${total}`;
      const qNote = query ? ` · query: <code>${query}</code>` : "";
      info.innerHTML =
        `<span class="good">${head} · ${total} rows${query ? " (filtered)" : ""} × ${curColumns.length} cols</span>${qNote}`;
      rerender();
      prevBtn.disabled = offset <= 0;
      nextBtn.disabled = to >= total;
      setStatus(`table loaded · ${total} rows (${r.source || "?"})`);
    } catch (e) {
      info.innerHTML = `<span class="error">${e.message}</span>`;
      setStatus(`table failed: ${e.message}`, "error");
    }
  }

  refreshBtn.addEventListener("click", () => { offset = 0; load(); });
  whichSel.addEventListener("change", () => { offset = 0; load(); });
  pageSizeSel.addEventListener("change", () => { offset = 0; load(); });
  applyBtn.addEventListener("click", () => { offset = 0; load(); });
  clearBtn.addEventListener("click", () => { queryInput.value = ""; offset = 0; load(); });
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { offset = 0; load(); }
  });
  prevBtn.addEventListener("click", () => {
    offset = Math.max(0, offset - parseInt(pageSizeSel.value));
    load();
  });
  nextBtn.addEventListener("click", () => {
    if (offset + parseInt(pageSizeSel.value) < total) {
      offset += parseInt(pageSizeSel.value);
      load();
    }
  });

  // ── notebook ─────────────────────────────────────────────────────────────
  const nbCells = root.querySelector("#nb-cells");
  const nbStatus = root.querySelector("#nb-status");
  const nbAddBtn = root.querySelector("#nb-add");
  const nbRunAllBtn = root.querySelector("#nb-run-all");
  const nbResetBtn = root.querySelector("#nb-reset");
  const cells = [];   // { el, code, runBtn, output }

  function _autosize(ta) {
    ta.style.height = "auto";
    ta.style.height = Math.min(400, Math.max(48, ta.scrollHeight)) + "px";
  }

  function _renderOutput(el, r) {
    el.innerHTML = "";
    const addPre = (text, cls) => {
      const p = document.createElement("pre");
      p.className = "nb-pre" + (cls ? " " + cls : "");
      p.textContent = text;
      el.appendChild(p);
    };
    if (r.error) addPre(r.error, "nb-err");
    if (r.stdout) addPre(r.stdout, "");
    if (r.stderr) addPre(r.stderr, "nb-warn");
    for (const img of (r.images || [])) {
      const im = document.createElement("img");
      im.className = "nb-img";
      im.src = `data:image/png;base64,${img}`;
      el.appendChild(im);
    }
    if (r.result != null) addPre(r.result, "nb-result");
    if (!r.error && !r.stdout && !r.stderr && r.result == null && !(r.images || []).length) {
      el.innerHTML = `<span class="muted">(no output)</span>`;
    }
  }

  async function _runCell(cell) {
    const code = cell.code.value;
    cell.runBtn.disabled = true;
    cell.output.innerHTML = `<span class="warn">running…</span>`;
    try {
      const r = await api.tableExec(code);
      _renderOutput(cell.output, r);
    } catch (e) {
      cell.output.innerHTML = `<pre class="nb-pre nb-err">${e.message}</pre>`;
    } finally {
      cell.runBtn.disabled = false;
    }
  }

  function addCell(code = "", focus = false) {
    const el = document.createElement("div");
    el.className = "nb-cell";
    el.innerHTML = `
      <div class="nb-input">
        <textarea class="nb-code" spellcheck="false" placeholder="df.describe()"></textarea>
        <div class="nb-cell-tools">
          <button class="nb-run mini">▶ Run</button>
          <button class="nb-del mini">✕</button>
        </div>
      </div>
      <div class="nb-output"></div>
    `;
    const ta = el.querySelector(".nb-code");
    const runBtn = el.querySelector(".nb-run");
    const delBtn = el.querySelector(".nb-del");
    const output = el.querySelector(".nb-output");
    ta.value = code;
    const cell = { el, code: ta, runBtn, output };
    cells.push(cell);
    nbCells.appendChild(el);

    ta.addEventListener("input", () => _autosize(ta));
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.shiftKey || e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        _runCell(cell);
        if (e.shiftKey && !e.ctrlKey && !e.metaKey) {
          const i = cells.indexOf(cell);
          if (i === cells.length - 1) addCell("", true);
          else cells[i + 1].code.focus();
        }
      } else if (e.key === "Tab") {
        // Insert two spaces instead of moving focus out of the cell.
        e.preventDefault();
        const s = ta.selectionStart, eend = ta.selectionEnd;
        ta.value = ta.value.slice(0, s) + "  " + ta.value.slice(eend);
        ta.selectionStart = ta.selectionEnd = s + 2;
      }
    });
    runBtn.addEventListener("click", () => _runCell(cell));
    delBtn.addEventListener("click", () => {
      const i = cells.indexOf(cell);
      if (i >= 0) cells.splice(i, 1);
      el.remove();
      if (!cells.length) addCell("", true);   // always keep one cell
    });

    _autosize(ta);
    if (focus) ta.focus();
    return cell;
  }

  async function runAll() {
    for (const cell of cells) await _runCell(cell);
  }

  async function resetKernel() {
    nbStatus.textContent = "resetting kernel…";
    try {
      const r = await api.tableKernelReset();
      const g4 = r.g4_rows != null
        ? `df_g4 = ${r.g4_rows} rows × ${r.g4_cols} cols`
        : "df_g4 = None";
      nbStatus.textContent = r.ok
        ? `kernel reset · df_sys = ${r.n_rows} rows × ${r.n_cols} cols · ${g4}`
        : `kernel: ${r.error}`;
    } catch (e) {
      nbStatus.textContent = `kernel reset failed: ${e.message}`;
    }
  }

  nbAddBtn.addEventListener("click", () => addCell("", true));
  nbRunAllBtn.addEventListener("click", runAll);
  nbResetBtn.addEventListener("click", resetKernel);

  // Seed one cell with a friendly example.
  addCell('df.describe()');

  await load();
}
