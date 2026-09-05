// Build a `processes` list from the global per-stage generator selections
// and the per-process ION checkbox in the top bar. Mirrors the shape
// `playground/services/sim_runner.py` expects; the presets it builds from
// are defined in `benchmark/track_simulator_config.py`.
//
// Shared by tabs/tracking.js (Trajectory + Pull) and tabs/pairwise.js — the
// per-stage globals + the SteppingManager schema are the single source of
// truth for what the simulator runs.
import { getAllStageGenerators, getAllProcesses } from "./state.js";

export function processesFromGlobals() {
  const g = getAllStageGenerators();
  const on = getAllProcesses();
  const list = [];

  if (on.ion) {
    const ion = { name: "g4h_ionisation" };
    if (g.length.kind === "nn") {
      ion.length_generator = { name: g.length.class_name, checkpoint: g.length.checkpoint };
      ion.length_x_scaler  = "LengthXScaler";
      ion.length_y_scaler  = "LengthYScaler";
    } else {
      ion.length_generator = { name: "PhysicsLengthGenerator" };
    }

    if (g.continuous.kind === "nn") {
      ion.continuous_generator = { name: g.continuous.class_name, checkpoint: g.continuous.checkpoint };
      ion.continuous_x_scaler  = "ContinuousXScaler";
      ion.continuous_y_scaler  = "ContinuousYScaler";
    } else {
      ion.continuous_generator = { name: "PhysicsContinuousGenerator" };
    }

    if (g.secondary.kind === "nn") {
      ion.secondary_generator = { name: g.secondary.class_name, checkpoint: g.secondary.checkpoint };
      ion.secondary_x_scaler  = "SecondaryXScaler";
      ion.secondary_y_scaler  = "SecondaryYScaler";
    } else {
      ion.secondary_generator = { name: "PhysicsSecondaryGenerator" };
    }
    list.push(ion);
  }

  return list;
}

// Human-readable blocker when the current selection can't form a meaningful
// simulation, or null when it's runnable. ION is the only process on this
// branch that samples an interaction length.
export function processSelectionError() {
  const on = getAllProcesses();
  if (!on.ion) {
    return "enable ION — no interaction-length sampler in the selection";
  }
  return null;
}
