// Entry point: load run context, build top selector bars, wire tab switching.
import { api, setStatus } from "./api.js";
import {
  setSelectorMode, setHalfwidthDex, setStageGenerator,
  getStageGenerator, STAGES, setParticle, setProcessEnabled,
} from "./state.js";
import { mountG4Browser }  from "./tabs/g4_browser.js";
import { mountGenerators } from "./tabs/generators.js";
import { mountTrajectory, mountPull } from "./tabs/tracking.js";
import { mountPairwise } from "./tabs/pairwise.js";
import { mountTable } from "./tabs/table.js";
import { mountCluster, refreshServer, shutdownServer } from "./tabs/cluster.js";

async function mountG4Combined(_root) {
  // The combined tab hosts two side-by-side mount points wired in index.html.
  const left  = document.getElementById("tab-g4-left");
  const right = document.getElementById("tab-g4-right");
  await mountG4Browser(left);
  await mountGenerators(right);
}

const mounters = {
  g4:         mountG4Combined,
  trajectory: mountTrajectory,
  pull:       mountPull,
  pairwise:   mountPairwise,
  table:      mountTable,
  cluster:    mountCluster,
};

const mounted = new Set();

function activateTab(tab) {
  document.querySelectorAll("nav.tabs button").forEach(b => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  document.querySelectorAll("main .tab").forEach(s => {
    s.classList.toggle("active", s.id === `tab-${tab}`);
  });
  if (!mounted.has(tab)) {
    mounted.add(tab);
    const el = document.getElementById(`tab-${tab}`);
    mounters[tab](el).catch(err => {
      el.innerHTML = `<div class="card error">Mount failed: ${err.message}</div>`;
      console.error(err);
    });
  }
}

function _wireSelectorMode() {
  document.querySelectorAll('input[name="ps-mode"]').forEach(r => {
    r.addEventListener("change", () => {
      if (r.checked) setSelectorMode(r.value);
    });
  });
  const hw = document.getElementById("ps-hw");
  hw.addEventListener("change", () => setHalfwidthDex(hw.value));
}

async function _wireGeneratorBar() {
  async function render(autoSelect) {
    let ckpts = { checkpoints: {} };
    try { ckpts = await api.checkpoints(); }
    catch (e) { setStatus(`generator checkpoints unavailable: ${e.message}`, "warn"); }

    for (const stage of STAGES) {
      const sel = document.getElementById(`gen-${stage}`);
      const prevValue = sel.value;
      sel.innerHTML = "";

      const mcOpt = document.createElement("option");
      mcOpt.value = "mc";
      mcOpt.textContent = "MC baseline (Physics sampler)";
      sel.appendChild(mcOpt);

      const stageCkpts = ckpts.checkpoints?.[stage] || [];
      for (const c of stageCkpts) {
        const o = document.createElement("option");
        const classes = ckpts.classes?.[stage] || [];
        let cls = classes[0];
        if (c.class_hint && classes.includes(c.class_hint)) {
          cls = c.class_hint;
        } else {
          for (const cand of classes) {
            if (c.filename.includes(cand) || c.run_id.includes(cand)) { cls = cand; break; }
          }
        }
        o.value = JSON.stringify({ kind: "nn", class_name: cls,
                                   checkpoint: c.path,
                                   run_id: c.run_id, filename: c.filename });
        o.textContent = `${cls} · ${c.run_id} · ${c.filename}`;
        o.dataset.path = c.path;
        sel.appendChild(o);
      }

      sel.removeEventListener("change", sel._handler || (() => {}));
      sel._handler = () => {
        if (sel.value === "mc") setStageGenerator(stage, { kind: "mc" });
        else setStageGenerator(stage, JSON.parse(sel.value));
      };
      sel.addEventListener("change", sel._handler);

      let restored = false;
      if (autoSelect && autoSelect.stage === stage) {
        for (const opt of sel.options) {
          if (opt.dataset.path === autoSelect.local_path) {
            sel.value = opt.value;
            sel.dispatchEvent(new Event("change"));
            restored = true;
            break;
          }
        }
      }
      if (!restored) {
        if (Array.from(sel.options).some(o => o.value === prevValue)) {
          sel.value = prevValue;
          sel.dispatchEvent(new Event("change"));
        } else {
          setStageGenerator(stage, { kind: "mc" });
        }
      }
    }
  }

  await render(null);
  window.refreshGeneratorBar = (importResult) => render(importResult);
}

// Process include/exclude checkboxes. Each checkbox toggles its whole group:
// the state flag consumed by processes.js plus a greyed-dropdowns class.
const PROCESS_IDS = ["ion"];

function _wireProcessCheckboxes(_particle) {
  for (const proc of PROCESS_IDS) {
    const box = document.getElementById(`proc-${proc}`);
    const grp = document.getElementById(`grp-${proc}`);
    const apply = () => {
      setProcessEnabled(proc, box.checked);
      grp.classList.toggle("off", !box.checked);
    };
    box.addEventListener("change", apply);
    apply();
  }
}

// Switch the active beam preset: POSTs to /api/run_context/set (which
// re-execs the Flask process with a new PHINGAN_BEAM env), then polls until
// the server answers again and hard-reloads the page so every cached
// module / UI state is rebuilt from scratch.
async function _switchBeam(beam) {
  setStatus(`switching to ${beam} — restarting server…`, "warn");
  try {
    await api.setRunContext({ beam });
  } catch (e) {
    // The server may close the connection mid-response if it re-execs fast;
    // that's expected, fall through to the polling loop below.
    console.warn("setRunContext threw (often expected during restart):", e);
  }
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 500));
    try {
      const ctx = await api.runContext();
      if (ctx.beam === beam) {
        setStatus(`switched to ${beam}, reloading…`, "good");
        location.reload();
        return;
      }
    } catch (_) { /* server still down, keep polling */ }
  }
  setStatus(`switch to ${beam} timed out — check server logs`, "error");
}

async function init() {
  setStatus("loading run context...");
  let particle = null;
  try {
    const ctx = await api.runContext();
    particle = ctx.particle;
    setParticle(particle);
    const choices = (ctx.available_beams || [ctx.beam]);
    // Beam dropdown: switching it POSTs to /api/run_context/set, which
    // re-execs the Python process with PHINGAN_BEAM in env. The reload is
    // necessary because run_context.ACTIVE / paths.* are read at import time
    // by every physics / scaler / checkpoint loader.
    const beamOpts = choices.map(b =>
      `<option value="${b}" ${b === ctx.beam ? "selected" : ""}>${b}</option>`
    ).join("");
    document.getElementById("run-context").innerHTML =
      `<label class="tag" style="display:inline-flex;align-items:center;gap:6px">` +
      `  beam <select id="rc-beam" style="background:#0e1116;color:#e6edf3;` +
      `    border:1px solid #30363d;border-radius:3px;padding:2px 4px">${beamOpts}</select>` +
      `</label>` +
      `<span class="tag">${ctx.particle}</span>` +
      `<span class="tag">${ctx.material}</span>` +
      `<span class="tag">${(ctx.beam_energy_eV / 1e6).toFixed(2)} MeV</span>` +
      `<span>tracks=${ctx.inventory.tracks.length}, cdfs=${ctx.inventory.cdfs.length}, ` +
      `scalers=${ctx.inventory.scalers.length}, datasets=${ctx.inventory.datasets.length}` +
      `${ctx.on_cluster ? ', cluster' : ', local'}</span>`;
    document.getElementById("rc-beam").addEventListener("change", (e) => {
      _switchBeam(e.target.value);
    });
    setStatus("ready");
  } catch (err) {
    setStatus(`run-context failed: ${err.message}`, "error");
  }

  _wireSelectorMode();
  await _wireGeneratorBar();
  _wireProcessCheckboxes(particle);

  document.querySelectorAll("nav.tabs button").forEach(b => {
    b.addEventListener("click", () => activateTab(b.dataset.tab));
  });

  document.getElementById("global-server-refresh")
    ?.addEventListener("click", () => refreshServer());
  document.getElementById("global-server-close")
    ?.addEventListener("click", () => shutdownServer());

  activateTab("g4");
}

init();
