import { DEFAULT_VERSION, MODELS, findModel } from "../models.js";
import { $, escapeHtml, say, toast } from "./dom.js";
import { DEV_FLAGS, EngineBusyError, HAS_WEBGPU, isCached, load, onEvict, unload } from "./engine.js";
import { createWebLLM, llm, mockLLM } from "./llm.js";
import { refreshAll } from "./render.js";
import { posts } from "./state.js";

// Start screen and model picker: which kobajinAI version writes the
// timeline.
//
// The page opens on a start screen (#gate), not on the timeline. Nothing
// generated is shown until a version is running, so a screenshot never
// shows canned text (AGENTS.md). A first-time visitor downloads nothing
// until they press the download button; the screen says first how much
// storage it needs and that 8 GB machines may struggle. Starting
// downloads that version once (about 2.2 GB) and keeps it in this
// browser; only the chosen version is downloaded and loaded.
//
// Once a version has come online here, the page remembers it
// (STARTED_KEY) and, on later visits, loads it again by itself if it is
// still saved in this browser: the start screen shows its progress.
// Stopping clears that and brings the start screen back.
//
// Only one tab runs the model (engine.js). Starting it here takes it from
// another tab, which goes back to the start screen ("elsewhere"); a
// later visit never takes it from another tab by itself.

const MODEL_KEY = "kobajinverse.model";
const STARTED_KEY = "kobajinverse.model-started";
const read = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
const write = (key, value) => { try { if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch { /* this visit only */ } };

// Free browser storage below this gets a warning: 2.2 GB of weights,
// WebLLM's library, and room to spare.
const NEEDED_BYTES = 3e9;
const GB = 1e9;

let chosen = findModel(read(MODEL_KEY) || DEFAULT_VERSION).version;
// checking | idle | loading | online | error | elsewhere | unsupported
let state = !HAS_WEBGPU ? "unsupported" : read(STARTED_KEY) ? "checking" : "idle";
let progress = { ratio: 0, text: "" };
let errorText = "";
let freeBytes = null; // browser storage left, when the browser says
const cached = new Set();

// Resolves the first time a version comes online; the timeline starts then.
let markReady;
export const modelReady = new Promise((resolve) => { markReady = resolve; });

const percent = () => `${Math.round(progress.ratio * 100)}%`;
const size = (bytes) => `約${(bytes / GB).toFixed(1)}GB`;

function chipText() {
  if (state === "online") return `kobajinAI ${chosen}`;
  if (state === "loading") return `読み込み中 ${percent()}`;
  return "停止中";
}

function stateText() {
  if (state === "online") return `kobajinAI ${chosen} がこの端末で動作中`;
  if (state === "loading") return `${chosen} を読み込み中… ${percent()}`;
  return "停止中";
}

function buttonText() {
  return {
    checking: "確認しています…",
    online: "停止する",
    loading: "中止",
    error: "もう一度試す",
    unsupported: "このブラウザでは使えません",
    elsewhere: `${chosen} をこのタブで起動`,
  }[state] || (cached.has(chosen) ? `${chosen} を起動` : `${chosen} をダウンロードして始める（約2.2GB）`);
}

function statusText() {
  return {
    checking: "このブラウザに保存済みのモデルを確認しています…",
    online: `${chosen} がこの端末の中で投稿・返信・リアクションを書いています。`,
    loading: progress.text,
    error: errorText,
    elsewhere: "kobajinAIは一度に1つのタブでだけ動きます（2つ動かすとメモリとGPUを2倍使うため）。起動すると、もう一方のタブは止まります。",
    unsupported: "WebGPUに対応したブラウザ（最新のChrome・Edge、Safari 26以降）で開くと、kobajinAIを動かせます。",
  }[state] || (cached.has(chosen)
    ? "このブラウザに保存済みなので、ダウンロードせずに起動できます。"
    : "Hugging Faceからモデルをダウンロードして、このブラウザに保存します。次からはダウンロードなしで起動します。");
}

// Storage and memory notes on the start screen. The memory note is always
// there; a device that reports less than 8 GB (navigator.deviceMemory,
// Chromium only, capped at 8) gets a stronger one.
function renderNotes() {
  const memory = navigator.deviceMemory;
  const lowMemory = typeof memory === "number" && memory < 8;
  $("gate-memory").textContent = lowMemory
    ? `この端末のメモリは${memory}GB程度のようです。kobajinAIはおそらく動きません。`
    : "メモリ8GBの端末では、重くなったり途中で止まったりすることがあります。ほかのタブやアプリを閉じてから始めてください。";
  $("gate-memory").closest("li").dataset.warn = String(lowMemory);
  const lowStorage = freeBytes !== null && freeBytes < NEEDED_BYTES && !cached.has(chosen);
  $("gate-storage").textContent = lowStorage
    ? `このブラウザで使える空きが${size(freeBytes)}しかありません。数GBの空きを作ってから始めてください。`
    : "端末のストレージに数GBの空きが必要です（モデルは約2.2GB）。";
  $("gate-storage").closest("li").dataset.warn = String(lowStorage);
}

function render() {
  document.body.dataset.model = state;
  document.body.dataset.gate = state === "online" ? "closed" : "open";
  $("model-state").textContent = stateText();
  $("model-chip").textContent = chipText();
  $("model-chip").dataset.state = state;

  const locked = state === "unsupported" || state === "checking";
  $("model-versions").innerHTML = MODELS.map((model) => `<label class="version">
      <input type="radio" name="model-version" value="${model.version}"${model.version === chosen ? " checked" : ""}${locked ? " disabled" : ""} />
      <span class="version-copy">
        <span class="version-name">${model.version}${model.version === DEFAULT_VERSION ? '<span class="badge alt">いちばんkobajinに近い</span>' : ""}${cached.has(model.version) ? '<span class="badge you">保存済み</span>' : ""}</span>
        <span class="version-note">${escapeHtml(model.note || model.summary)}</span>
      </span>
    </label>`).join("");
  $("gate-version").innerHTML = MODELS.map((model) => `<option value="${model.version}"${model.version === chosen ? " selected" : ""}>${model.version}${model.version === DEFAULT_VERSION ? "（いちばんkobajinに近い）" : ""}${cached.has(model.version) ? "・保存済み" : ""}</option>`).join("");
  $("gate-version").disabled = locked || state === "loading";

  const action = state === "online" || state === "loading" ? "stop" : "start";
  for (const [button, status, bar, box] of [
    [$("model-start"), $("model-status"), $("model-progress-bar"), $("model-progress")],
    [$("gate-start"), $("gate-status"), $("gate-progress-bar"), $("gate-progress")],
  ]) {
    button.disabled = locked;
    button.dataset.action = action;
    button.textContent = buttonText();
    status.textContent = statusText();
    box.hidden = state !== "loading";
    bar.style.width = percent();
  }
  $("gate-start").classList.toggle("quiet", state === "loading");
  renderNotes();
}

function goOnline(engine, version) {
  llm.use(createWebLLM(engine, version, (error) => {
    // The engine broke (GPU lost, tab out of memory): back to the start
    // screen. Not online means it was being unloaded on purpose (stop,
    // switch, another tab), which cuts a running generation short.
    if (state !== "online") return;
    console.error("kobajinAI generation failed", error);
    llm.use(null);
    unload();
    state = "error";
    errorText = "生成中にエラーが起きたため停止しました。ほかのタブを閉じてから、もう一度試してください。";
    render();
  }));
  state = "online";
  cached.add(version);
  write(STARTED_KEY, "1");
  render();
  refreshAll(posts);
  markReady();
  say(`kobajinAI ${version} がオンラインになりました`);
  toast(`kobajinAI ${version} が起動しました`);
}

// Progress reports come many times a second; repaint once per frame.
let renderQueued = false;
function renderSoon() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; render(); });
}

// `steal` takes the model from another tab; without it, a tab that
// finds the model running elsewhere waits on the start screen.
async function start({ steal = true } = {}) {
  if (!HAS_WEBGPU) return;
  const version = chosen;
  llm.use(null);
  state = "loading";
  progress = { ratio: 0, text: "準備しています…" };
  render();
  try {
    const loaded = await load(version, (report) => {
      progress = { ratio: Math.max(progress.ratio, report.progress || 0), text: describe(report.text) };
      renderSoon();
    }, { steal });
    if (!loaded) return; // superseded by another start or a stop
    goOnline(loaded.engine, loaded.version);
  } catch (error) {
    if (error instanceof EngineBusyError) {
      state = "elsewhere";
      render();
      return;
    }
    console.error("kobajinAI model load failed", error);
    state = "error";
    errorText = `${error?.message || "読み込みに失敗しました"}。通信とストレージの空き容量を確かめて、もう一度試してください。`;
    render();
  }
}

async function stop() {
  write(STARTED_KEY, null);
  llm.use(null);
  state = "idle";
  $("model-dialog").close();
  render();
  await unload();
}

// WebLLM's progress text is English; say what step it is in.
function describe(text = "") {
  if (/Fetching param cache|Loading model from cache|fetching/i.test(text)) {
    const done = /(\d+)MB fetched/.exec(text);
    return done ? `モデルを取得中（${done[1]}MB）…` : "モデルを取得中…";
  }
  if (/Loading GPU shader|shader/i.test(text)) return "GPUの準備中…";
  if (/Finish/i.test(text)) return "仕上げ中…";
  return "読み込み中…";
}

// Another tab started the model and took it from this one.
function evicted() {
  if (state !== "online" && state !== "loading") return;
  llm.use(null);
  state = "elsewhere";
  render();
  toast("別のタブでkobajinAIを起動したので、このタブは止まりました");
}

function choose(version) {
  chosen = findModel(version).version;
  write(MODEL_KEY, chosen);
  // Switching while running loads the new version in its place.
  if (state === "online" || state === "loading") start(); else { if (state === "error") state = "idle"; render(); }
}

// Finds which versions are saved here and how much storage is left, then
// brings back a version this visitor ran before.
async function survey() {
  await Promise.all([
    ...MODELS.map(async (model) => { if (await isCached(model.version)) cached.add(model.version); }),
    navigator.storage?.estimate?.().then(({ quota, usage }) => {
      if (quota) freeBytes = Math.max(0, quota - (usage || 0));
    }).catch(() => {}),
  ]);
  if (state === "checking") {
    if (cached.has(chosen)) { start({ steal: false }); return; }
    state = "idle";
  }
  if (state !== "online" && state !== "loading") render();
}

export function setupModelPicker() {
  // Development only: run on the mock without the start screen.
  if (import.meta.env.DEV && DEV_FLAGS.has("mock")) {
    llm.use(mockLLM());
    document.body.dataset.gate = "closed";
    document.body.dataset.model = "mock";
    $("model-chip").textContent = "モック";
    $("model-state").textContent = "モック推論（開発用）";
    markReady();
    return;
  }
  onEvict(evicted);
  const dialog = $("model-dialog");
  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-open-model]")) { render(); dialog.showModal(); }
  });
  dialog.addEventListener("click", (event) => {
    if (event.target.closest("[data-close]") || event.target === dialog) dialog.close();
  });
  $("model-versions").addEventListener("change", (event) => {
    if (event.target.name === "model-version") choose(event.target.value);
  });
  $("gate-version").addEventListener("change", (event) => choose(event.target.value));
  for (const button of [$("model-start"), $("gate-start")]) {
    button.addEventListener("click", () => {
      if (button.dataset.action === "stop") stop(); else start();
    });
  }
  render();
  if (HAS_WEBGPU) survey().catch(() => { if (state === "checking") { state = "idle"; render(); } });
}
