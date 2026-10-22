"""Serve the precomputed one-case Beta 1 professor demo on localhost."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "outputs" / "beta1_professor_demo" / "web"
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CancerAgents Beta 1 demonstration</title>
<style>
  :root { color-scheme: light; --ink:#1d1f23; --muted:#5d646e; --line:#d7dbe1; --paper:#f6f7f8; --accent:#1f4e79; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.45 "Source Sans 3", "Segoe UI", sans-serif; color:var(--ink); background:var(--paper); }
  header, main { width:min(1180px, calc(100% - 32px)); margin:0 auto; }
  header { padding:22px 0 8px; }
  h1 { margin:0 0 4px; font-size:24px; letter-spacing:0; }
  h2 { margin:22px 0 8px; font-size:16px; }
  p, li { color:var(--muted); margin:6px 0; }
  .banner { display:flex; justify-content:space-between; gap:16px; border-top:3px solid var(--accent); padding-top:12px; }
  .status { font-size:12px; letter-spacing:.04em; text-transform:uppercase; color:var(--accent); }
  .controls, .metrics, .note { background:white; border:1px solid var(--line); padding:14px; }
  .controls { display:grid; grid-template-columns:1.3fr .9fr .8fr; gap:18px; margin-top:14px; }
  label { display:flex; align-items:center; gap:8px; margin:4px 0; }
  button, input { font:inherit; }
  button { border:1px solid var(--line); background:white; padding:5px 9px; }
  button[aria-pressed="true"] { border-color:var(--accent); color:var(--accent); }
  .row { display:flex; flex-wrap:wrap; gap:6px; }
  .stage { margin-top:14px; background:white; border:1px solid var(--line); padding:12px; }
  .frame { position:relative; width:min(620px, 100%); margin:0 auto; background:#111; }
  .frame img, .grid img { width:100%; height:auto; image-rendering:pixelated; display:block; }
  .overlay { position:absolute; inset:0; pointer-events:none; }
  .grid { display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; }
  figure { margin:0; padding:4px; }
  figure.selected { outline:3px solid var(--accent); }
  figcaption { font-size:12px; margin-top:4px; color:var(--ink); }
  table { width:100%; border-collapse:collapse; background:white; }
  th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }
  td:not(:first-child), th:not(:first-child) { text-align:right; font-variant-numeric:tabular-nums; }
  .swatch { width:11px; height:11px; display:inline-block; margin-right:6px; vertical-align:-1px; }
  @media (max-width:900px) { .controls, .grid { grid-template-columns:1fr 1fr; } }
</style>
</head>
<body>
<header>
  <div class="banner">
    <div>
      <div class="status">Research demonstration · precomputed · not for diagnosis</div>
      <h1>CancerAgents Beta 1</h1>
      <p id="case-line">Loading frozen validation case…</p>
    </div>
    <p id="slice-readout"></p>
  </div>
</header>
<main>
  <section class="controls">
    <div>
      <strong>MRI modality</strong>
      <div class="row" id="modalities"></div>
      <label>Slice <input id="slice" type="range" min="0" value="0"></label>
      <div class="row">
        <button id="single" aria-pressed="true" type="button">Single view</button>
        <button id="compare" aria-pressed="false" type="button">Expert comparison</button>
      </div>
    </div>
    <div>
      <strong>Segmentation overlay</strong>
      <div id="overlays"></div>
      <label>Opacity <input id="opacity" type="range" min="0" max="100" value="58"></label>
      <p id="overlay-status"></p>
    </div>
    <div>
      <strong>Label colors</strong>
      <p><span class="swatch" style="background:#d5289b"></span>Enhancing tumor, label 3. This is ET.</p>
      <p><span class="swatch" style="background:#e66101"></span>Non-enhancing core, label 1. Tumor core is labels 1 and 3.</p>
      <p><span class="swatch" style="background:#5eac43"></span>Edema, label 2. Whole tumor is labels 1, 2, and 3.</p>
    </div>
  </section>
  <section class="stage" id="single-stage">
    <div class="frame">
      <img id="mri" alt="Selected MRI slice">
      <img class="overlay" id="mask" alt="">
    </div>
  </section>
  <section class="stage" id="compare-stage" hidden>
    <div class="grid" id="grid"></div>
  </section>
  <h2>Held-out test summary</h2>
  <table id="metrics"></table>
  <p>These are aggregate scores for 130 previously evaluated test cases. They are not the Dice of the validation patient shown above. The controller mean is 0.00032 above majority voting, so this demonstration does not claim a substantial improvement over the vote.</p>
  <section class="note">
    <h2>HighRes distinction</h2>
    <p id="highres-note"></p>
  </section>
</main>
<script>
const state = { meta:null, modality:"t2f", overlay:"final", opacity:0.58, slice:0, mode:"single" };
const compareIds = ["mri","cnn","transformer","boundary","highres","highres_full","final","vote","ground_truth","disagreement"];
function slicePath(kind, id, index) {
  const folder = kind === "mri" ? `slices/mri/${id}` : `slices/seg/${id}`;
  return `${folder}/${String(index).padStart(3,"0")}.png?v=${state.meta.asset_version || 3}`;
}
function describeOverlay() {
  const status = document.getElementById("overlay-status");
  let detail = "MRI only. No segmentation is drawn.";
  if (state.overlay !== "none") {
    const label = state.meta.overlays.find(item => item.id === state.overlay).label;
    if (state.overlay === "disagreement") detail = `${label}. Yellow shows where the experts disagree. It is not a tumor label.`;
    else if (state.overlay === "ground_truth") detail = "Ground truth. This is the reference, so no cyan difference marks are drawn.";
    else {
      const counts = state.meta.slice_differences && state.meta.slice_differences[state.overlay];
      const count = counts ? counts[state.slice] : null;
      detail = count == null ? label : `${label}. Cyan marks ${count.toLocaleString()} pixels on this slice that differ from ground truth.`;
    }
  }
  status.textContent = state.mode === "compare"
    ? `${detail} Expert comparison keeps every mask visible; the selected one is outlined.`
    : detail;
}
function renderReadout() {
  document.getElementById("slice-readout").textContent = `Native axis 2 · slice ${state.slice} of ${state.meta.slice_count - 1}`;
}
function renderSingle() {
  const mri = document.getElementById("mri");
  const mask = document.getElementById("mask");
  mri.src = slicePath("mri", state.modality, state.slice);
  mri.alt = `${state.modality} slice ${state.slice}`;
  if (state.overlay === "none") { mask.hidden = true; describeOverlay(); return; }
  mask.hidden = false;
  mask.src = slicePath("seg", state.overlay, state.slice);
  mask.style.opacity = state.overlay === "disagreement" ? 1 : state.opacity;
  describeOverlay();
}
function renderGrid() {
  const grid = document.getElementById("grid");
  grid.replaceChildren();
  for (const id of compareIds) {
    const match = id === "mri"
      ? state.meta.modalities.find(item => item.id === state.modality)
      : state.meta.overlays.find(item => item.id === id);
    const spec = match || {label: id};
    const figure = document.createElement("figure");
    const selectedId = state.overlay === "none" ? "mri" : state.overlay;
    if (id === selectedId) figure.className = "selected";
    const frame = document.createElement("div");
    frame.className = "frame";
    const base = document.createElement("img");
    base.src = slicePath("mri", state.modality, state.slice);
    base.alt = `${spec.label}, slice ${state.slice}`;
    frame.append(base);
    if (id !== "mri") {
      const overlay = document.createElement("img");
      overlay.className = "overlay";
      overlay.alt = "";
      overlay.src = slicePath("seg", id, state.slice);
      overlay.style.opacity = id === "disagreement" ? 1 : state.opacity;
      frame.append(overlay);
    }
    const caption = document.createElement("figcaption");
    caption.textContent = spec.label;
    figure.append(frame, caption);
    grid.append(figure);
  }
}
function render(options = {}) {
  renderReadout();
  renderSingle();
  if (state.mode === "compare") {
    renderGrid();
    if (options.focusSelected) {
      const selected = document.querySelector("#grid figure.selected");
      if (selected) selected.scrollIntoView({behavior:"smooth", block:"nearest"});
    }
  }
}
function selectedOverlay() {
  const chosen = document.querySelector("input[name=overlay]:checked");
  state.overlay = chosen ? chosen.value : "none";
}
async function init() {
  const response = await fetch("meta.json");
  state.meta = await response.json();
  document.getElementById("case-line").textContent = `${state.meta.patient_id} · frozen validation case · predictions were precomputed with frozen Beta 1 checkpoints`;
  state.slice = state.meta.default_slice;
  const slider = document.getElementById("slice");
  slider.max = state.meta.slice_count - 1;
  slider.value = state.slice;
  const modalities = document.getElementById("modalities");
  for (const item of state.meta.modalities) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.label;
    button.setAttribute("aria-pressed", String(item.id === state.modality));
    button.addEventListener("click", () => {
      state.modality = item.id;
      for (const child of modalities.children) child.setAttribute("aria-pressed", String(child === button));
      render();
    });
    modalities.append(button);
  }
  const overlays = document.getElementById("overlays");
  for (const item of [{id:"none", label:"MRI only"}, ...state.meta.overlays]) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "overlay";
    input.value = item.id;
    input.checked = item.id === state.overlay;
    input.addEventListener("change", () => { selectedOverlay(); render({focusSelected:true}); });
    label.append(input, document.createTextNode(item.label));
    overlays.append(label);
  }
  slider.addEventListener("input", () => { state.slice = Number(slider.value); render(); });
  document.getElementById("opacity").addEventListener("input", event => { state.opacity = Number(event.target.value) / 100; render(); });
  document.getElementById("single").addEventListener("click", () => setMode("single"));
  document.getElementById("compare").addEventListener("click", () => setMode("compare"));
  document.addEventListener("keydown", event => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const delta = event.key === "ArrowRight" ? 1 : -1;
    state.slice = Math.min(state.meta.slice_count - 1, Math.max(0, state.slice + delta));
    slider.value = state.slice;
    render();
  });
  const metrics = state.meta.test_metrics.mean_dice;
  const rows = [["CNN", metrics.cnn], ["Transformer", metrics.transformer], ["Boundary", metrics.boundary], ["HighRes, full-volume protocol", metrics.highres_full_volume], ["Controller, epoch 19", metrics.controller], ["Majority voting", metrics.majority_vote]];
  document.getElementById("metrics").innerHTML = `<tr><th>Method</th><th>Mean Dice</th></tr>` + rows.map(([name, value]) => `<tr><td>${name}</td><td>${value.toFixed(5)}</td></tr>`).join("");
  document.getElementById("highres-note").textContent = `Proposal-limited coverage on this case is ${(100 * state.meta.proposal_coverage).toFixed(1)}% of voxels. ${state.meta.highres_voxels_different.toLocaleString()} voxels differ from the separate full-volume HighRes pass. ${state.meta.highres_note}`;
  render();
}
function setMode(mode) {
  state.mode = mode;
  document.getElementById("single").setAttribute("aria-pressed", String(mode === "single"));
  document.getElementById("compare").setAttribute("aria-pressed", String(mode === "compare"));
  document.getElementById("single-stage").hidden = mode !== "single";
  document.getElementById("compare-stage").hidden = mode !== "compare";
  render();
  document.getElementById(mode === "compare" ? "compare-stage" : "single-stage").scrollIntoView({behavior:"smooth", block:"start"});
}
init();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not (WEB_DIR / "meta.json").is_file():
        raise SystemExit("Precomputed demo assets are missing. Run scripts/beta1_professor_demo.py first.")
    handler = partial(SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    page_path = WEB_DIR / "index.html"
    page_path.write_text(PAGE)
    url = f"http://{args.host}:{args.port}/"
    print(url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
