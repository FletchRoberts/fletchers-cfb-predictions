// Fletcher's CFB Predictions -- shared helpers

const DATA_BASE = "data/";

function fmtSigned(n) {
  const v = Number(n);
  return (v >= 0 ? "+" : "") + v.toFixed(1);
}

async function loadJSON(name) {
  const res = await fetch(DATA_BASE + name, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load " + name + " (" + res.status + ")");
  return res.json();
}

function renderUpdatedBadge(elementId, meta) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.textContent = meta.season + " SEASON \u00b7 UPDATED " + meta.updated_at +
    " \u00b7 " + meta.games_processed + " GAMES PROCESSED";
}
