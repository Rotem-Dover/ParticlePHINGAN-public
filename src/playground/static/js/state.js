// Shared global state for the playground UI.
//
// Three orthogonal pieces of state live here:
//   1. Phase-space location: (E_eV, L_m)        — pinned from the G4 browser.
//   2. Selector mode:        'bin' | 'point'    + halfwidth (dex) when bin.
//      - 'bin'   ⇒ downstream plots include G4 truth restricted to the cell
//      - 'point' ⇒ downstream plots show only sampling/PDF curves (no G4)
//   3. Per-stage generator choice for each of length/continuous/secondary:
//        {kind: 'mc'}                            — analytical Physics MC sampler
//        {kind: 'nn', class_name, checkpoint, run_id, filename}
//      Default is 'mc' for every stage.
//
// All listeners get the full state snapshot. Combining the three streams into
// a single onChange channel keeps subscribers simple (e.g. the generators tab
// needs to react to any of them).

const STAGES = ["length", "continuous", "secondary"];
const PROCESSES = ["ion"];

const listeners = new Set();
const state = {
  E_eV: null,
  L_m:  null,
  mode: "point",          // 'bin' | 'point'
  halfwidth_dex: 0.05,
  generators: Object.fromEntries(STAGES.map(s => [s, { kind: "mc" }])),
  // Per-process include/exclude (the top-bar checkboxes). All on by default.
  processes: Object.fromEntries(PROCESSES.map(p => [p, true])),
  particle: null,         // active particle name from /api/run_context
};

function _notify() {
  const snap = getState();
  listeners.forEach(fn => { try { fn(snap); } catch (e) { console.error(e); }});
}

export function getState() {
  return JSON.parse(JSON.stringify(state));   // defensive clone
}

export function setPhaseSpacePoint(E_eV, L_m) {
  state.E_eV = E_eV;
  state.L_m  = L_m;
  _updateBanner();
  _notify();
}

export function getPhaseSpacePoint() {
  return { E_eV: state.E_eV, L_m: state.L_m };
}

export function setSelectorMode(mode) {
  if (mode !== "bin" && mode !== "point") return;
  state.mode = mode;
  _updateBanner();
  _notify();
}

export function getSelectorMode()    { return state.mode; }
export function setHalfwidthDex(hw)  { state.halfwidth_dex = Number(hw); _notify(); }
export function getHalfwidthDex()    { return state.halfwidth_dex; }

export function setStageGenerator(stage, choice) {
  if (!STAGES.includes(stage)) return;
  state.generators[stage] = choice;
  _notify();
}

export function getStageGenerator(stage)   { return state.generators[stage]; }
export function getAllStageGenerators()    { return { ...state.generators }; }

export function setProcessEnabled(proc, on) {
  if (!PROCESSES.includes(proc)) return;
  state.processes[proc] = !!on;
  _notify();
}

export function getProcessEnabled(proc) { return state.processes[proc]; }
export function getAllProcesses()       { return { ...state.processes }; }

export function setParticle(p) { state.particle = p; }
export function getParticle()  { return state.particle; }

export function onChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function _updateBanner() {
  const banner = document.getElementById("ps-summary");
  if (!banner) return;
  const e = state.E_eV != null ? `E = ${formatEng(state.E_eV)} eV` : "E = —";
  const l = state.L_m  != null ? `L = ${formatEng(state.L_m)} m`   : "L = —";
  const tag = state.mode === "bin"
    ? `[bin · ±${state.halfwidth_dex} dex]`
    : "[point]";
  banner.textContent = `${e},  ${l}   ${tag}`;
}

export function formatEng(v) {
  if (v == null || !isFinite(v)) return "—";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  const exp = Math.floor(Math.log10(abs));
  const mant = v / Math.pow(10, exp);
  return `${mant.toFixed(3)}e${exp >= 0 ? "+" : ""}${exp}`;
}

export { STAGES, PROCESSES };
