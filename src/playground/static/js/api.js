// Tiny fetch wrapper. All endpoints live under /api and return JSON.
// Backend errors are propagated as Error("status code: message").

async function _request(url, opts = {}) {
  const res = await fetch(url, opts);
  let body;
  try { body = await res.json(); }
  catch { body = { error: `non-JSON response (${res.status})` }; }
  if (!res.ok) throw new Error(`${res.status}: ${body.error || JSON.stringify(body)}`);
  return body;
}

export const api = {
  runContext: () => _request("/api/run_context/"),
  setRunContext: (payload) => _request("/api/run_context/set", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload),
  }),

  g4Files:    () => _request("/api/g4/files"),
  g4Summary:  (file) => _request(`/api/g4/summary?file=${encodeURIComponent(file)}`),
  g4Hist2d:   (params) => _request(`/api/g4/hist2d?${new URLSearchParams(params)}`),
  g4PhaseSpaceRegions: () => _request("/api/g4/phase_space_regions"),
  g4Hist1d:   (params) => _request(`/api/g4/hist1d?${new URLSearchParams(params)}`),

  samplerStages: () => _request("/api/samplers/stages"),
  sample:     (params) => _request(`/api/samplers/sample?${new URLSearchParams(params)}`),
  pdfCurve:   (params) => _request(`/api/pdfs/curve?${new URLSearchParams(params)}`),

  checkpoints: () => _request("/api/generators/checkpoints"),
  generate:   (params) => _request(`/api/generators/generate?${new URLSearchParams(params)}`),

  presets:    () => _request("/api/stepping/presets"),
  simProgress:() => _request("/api/stepping/progress"),
  runSim:     (payload) => _request("/api/stepping/run", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload),
  }),

  pull:       (payload) => _request("/api/pull/run", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload),
  }),
  rebin:      (payload) => _request("/api/pull/rebin", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload),
  }),

  pairwiseFeatures: () => _request("/api/pairwise/features"),
  pairwise:   (payload) => _request("/api/pairwise/run", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload),
  }),

  tableLast:  (params = {}) => _request(`/api/table/last?${new URLSearchParams(params)}`),
  tableExec:  (code) => _request("/api/table/exec", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ code }),
  }),
  tableKernelReset: () => _request("/api/table/kernel/reset", { method: "POST" }),

  clusterPing:   ()      => _request("/api/cluster/ping"),
  clusterLs:     (path)  => _request(`/api/cluster/ls?path=${encodeURIComponent(path)}`),
  clusterImport: (path)  => _request("/api/cluster/import", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({path}),
  }),

  adminHealth:   ()      => _request("/api/admin/health"),
  adminRefresh: ()       => _request("/api/admin/refresh", {method: "POST"}),
  adminShutdown:()       => _request("/api/admin/shutdown", {method: "POST"}),
};

export function setStatus(text, level = "info") {
  const el = document.getElementById("status-bar");
  if (!el) return;
  el.textContent = text || "";
  el.className = level;
}
