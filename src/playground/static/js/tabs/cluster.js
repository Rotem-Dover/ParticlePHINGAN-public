// Cluster tab: connection light + Check / Refresh / Close server buttons,
// remote tensorboard browser, status line.

import { api, setStatus } from "../api.js";

const STATUS_LIGHT_COLOR = {unknown: "#666", ok: "#3fb950", down: "#f97583"};

function setLight(el, state, tooltip) {
  el.style.background = STATUS_LIGHT_COLOR[state] || STATUS_LIGHT_COLOR.unknown;
  el.title = tooltip || state;
}

async function checkConnection(light, statusEl) {
  setLight(light, "unknown", "checking…");
  try {
    const res = await api.clusterPing();
    if (res.ok) {
      setLight(light, "ok", `gpu2 reachable · ${res.latency_ms} ms`);
      statusEl.textContent = `connection ok (${res.latency_ms} ms)`;
    } else {
      setLight(light, "down", res.error || "unreachable");
      statusEl.textContent = `connection failed: ${res.error}`;
    }
  } catch (e) {
    setLight(light, "down", e.message);
    statusEl.textContent = `connection failed: ${e.message}`;
  }
}

export async function refreshServer() {
  if (!confirm("Restart the playground server?")) return;
  showOverlay("Restarting playground server…");
  try { await api.adminRefresh(); } catch (_) { /* expected mid-restart */ }
  await waitForHealth(/*reload=*/true);
}

export async function shutdownServer() {
  if (!confirm("Stop the playground server? You will need to relaunch it from a terminal.")) return;
  showOverlay("Server stopped — close this tab.", /*spinner=*/false);
  try { await api.adminShutdown(); } catch (_) { /* expected */ }
}

function showOverlay(text, spinner = true) {
  let el = document.getElementById("admin-overlay");
  if (!el) {
    el = document.createElement("div");
    el.id = "admin-overlay";
    el.style.cssText = "position:fixed;inset:0;background:rgba(13,17,23,0.92);" +
      "color:#e6edf3;display:flex;align-items:center;justify-content:center;" +
      "font-size:18px;z-index:9999;flex-direction:column;gap:12px";
    document.body.appendChild(el);
  }
  el.innerHTML = (spinner ? "<div>⏳</div>" : "") + `<div>${text}</div>`;
}

async function waitForHealth(reloadOnSuccess) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 500));
    try {
      const h = await api.adminHealth();
      if (h.ok) {
        if (reloadOnSuccess) location.reload();
        return true;
      }
    } catch (_) { /* keep polling */ }
  }
  showOverlay("Restart took too long — check the terminal.", false);
  return false;
}

const treeCache = new Map();

function sortEntries(entries) {
  return entries.slice().sort((a, b) => {
    // last.ckpt always on top within its folder.
    if (a.name === "last.ckpt" && b.name !== "last.ckpt") return -1;
    if (b.name === "last.ckpt" && a.name !== "last.ckpt") return 1;
    // epoch=N.ckpt sorted by N descending.
    const ea = a.name.match(/^epoch=(\d+)\.ckpt$/);
    const eb = b.name.match(/^epoch=(\d+)\.ckpt$/);
    if (ea && eb) return parseInt(eb[1], 10) - parseInt(ea[1], 10);
    if (ea) return -1;
    if (eb) return 1;
    // otherwise: directories first, then alphabetical.
    if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

async function fetchEntries(path) {
  if (treeCache.has(path)) return treeCache.get(path);
  const {entries} = await api.clusterLs(path);
  const sorted = sortEntries(entries);
  treeCache.set(path, sorted);
  return sorted;
}

function makeFileRow(entry, parentPath, onImport) {
  const li = document.createElement("li");
  const size = entry.size_mb != null ? `${entry.size_mb} MB` : "";
  li.innerHTML = `<span>${entry.name}</span> <span style="color:#888">${size}</span>`;
  if (entry.name.endsWith(".ckpt")) {
    const btn = document.createElement("button");
    btn.textContent = "Import";
    btn.style.marginLeft = "8px";
    btn.addEventListener("click", () => onImport(`${parentPath}${parentPath ? "/" : ""}${entry.name}`));
    li.appendChild(btn);
  }
  return li;
}

async function makeFolderRow(entry, parentPath, onImport, statusEl) {
  const details = document.createElement("details");
  details.dataset.path = `${parentPath}${parentPath ? "/" : ""}${entry.name}`;
  const summary = document.createElement("summary");
  summary.textContent = entry.name + "/";
  details.appendChild(summary);
  const inner = document.createElement("ul");
  inner.style.cssText = "list-style:none;padding-left:18px";
  details.appendChild(inner);

  let loaded = false;
  details.addEventListener("toggle", async () => {
    if (!details.open || loaded) return;
    inner.innerHTML = "<li>loading…</li>";
    try {
      const entries = await fetchEntries(details.dataset.path);
      inner.innerHTML = "";
      for (const e of entries) {
        if (e.is_dir) inner.appendChild(await makeFolderRow(e, details.dataset.path, onImport, statusEl));
        else          inner.appendChild(makeFileRow(e, details.dataset.path, onImport));
      }
      loaded = true;
    } catch (err) {
      inner.innerHTML = `<li style="color:#f97583">${err.message}</li>`;
    }
  });
  return details;
}

async function renderTree(root, statusEl, onImport) {
  root.innerHTML = "loading…";
  try {
    const entries = await fetchEntries("");
    const ul = document.createElement("ul");
    ul.style.cssText = "list-style:none;padding-left:0";
    for (const e of entries) {
      if (e.is_dir) ul.appendChild(await makeFolderRow(e, "", onImport, statusEl));
      else          ul.appendChild(makeFileRow(e, "", onImport));
    }
    root.innerHTML = "";
    root.appendChild(ul);
  } catch (err) {
    root.innerHTML = `<div style="color:#f97583">cluster unreachable — check connection (${err.message})</div>`;
  }
}

async function handleImport(relPath, statusEl, onImported) {
  statusEl.textContent = `importing ${relPath}…`;
  try {
    const res = await api.clusterImport(relPath);
    statusEl.textContent = `imported ${relPath} → ${res.size_mb} MB`;
    onImported(res);
  } catch (err) {
    statusEl.textContent = `import failed: ${err.message}`;
  }
}

export async function mountCluster(root) {
  root.innerHTML = `
    <div class="cluster-admin-strip" style="display:flex;gap:12px;align-items:center;padding:8px 12px;border-bottom:1px solid #30363d">
      <span id="cluster-light"
            style="width:12px;height:12px;border-radius:50%;background:#666;display:inline-block"></span>
      <span>cluster gpu2</span>
      <button id="cluster-check">Check connection</button>
    </div>
    <div id="cluster-tree" style="padding:12px"></div>
    <div id="cluster-status" style="padding:8px 12px;color:#888"></div>
  `;
  const light = root.querySelector("#cluster-light");
  const tree = root.querySelector("#cluster-tree");
  const statusEl = root.querySelector("#cluster-status");

  const onImported = async (importResult) => {
    if (window.refreshGeneratorBar) {
      await window.refreshGeneratorBar(importResult);
    }
  };

  const onImport = (path) => handleImport(path, statusEl, onImported);

  root.querySelector("#cluster-check").addEventListener("click", async () => {
    await checkConnection(light, statusEl);
    treeCache.clear();
    await renderTree(tree, statusEl, onImport);
  });

  await checkConnection(light, statusEl);
  await renderTree(tree, statusEl, onImport);
}
