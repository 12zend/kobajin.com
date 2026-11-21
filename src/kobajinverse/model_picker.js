import { DEFAULT_VERSION, MODELS, findModel } from "../models.js";
import { $, escapeHtml, say, toast } from "./dom.js";
import { EngineBusyError, HAS_WEBGPU, isCached, load, onEvict, unload } from "./engine.js";
import { createWebLLM, llm } from "./llm.js";
import { refreshAll } from "./render.js";
import { posts } from "./state.js";

// Model picker: which kobajinAI version writes the timeline.
//
// The page runs on the mock until the visitor starts a version. Starting
// downloads that version once (about 2.2 GB) and keeps it in this
// browser; only the chosen version is downloaded and loaded (AGENTS.md).
// A visitor who started the model before, and whose version is saved
// here, gets it back as soon as they post (wakeModel).
//
// Only one tab runs the model (engine.js). Starting it here takes it from
// another tab, which goes back to the mock ("elsewhere"); posting in a
// tab never takes it from another one.

const MODEL_KEY = "kobajinverse.model";
const STARTED_KEY = "kobajinverse.model-started";
const read = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
const write = (key, value) => { try { if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch { /* this visit only */ } };

let chosen = findModel(read(MODEL_KEY) || DEFAULT_VERSION).version;
// idle | loading | online | error | elsewhere | unsupported
let state = HAS_WEBGPU ? "idle" : "unsupported";
let progress = { ratio: 0, text: "" };
let errorText = "";
const cached = new Set();

function chipText() {
  if (state === "online") return `kobajinAI ${chosen}`;
  if (state === "loading") return `読み込み中 ${Math.round(progress.ratio * 100)}%`;
  return "モック";
}

function stateText() {
  if (state === "online") return `kobajinAI ${chosen} がこの端末で動作中`;
  if (state === "loading") return `${chosen} を読み込み中… ${Math.round(progress.ratio * 100)}%`;
  if (state === "error") return "読み込めませんでした（モック推論で動作中）";
  if (state === "elsewhere") return "別のタブで動作中（このタブはモック推論）";
  if (state === "unsupported") return "モック推論（WebGPU非対応）";
  return "モック推論で動作中";
}

function render() {
  document.body.dataset.model = state;
  $("model-state").textContent = stateText();
  $("model-chip").textContent = chipText();
  $("model-chip").dataset.state = state;

  $("model-versions").innerHTML = MODELS.map((model) => `<label class="version">
      <input type="radio" name="model-version" value="${model.version}"${model.version === chosen ? " checked" : ""}${state === "unsupported" ? " disabled" : ""} />
      <span class="version-copy">
        <span class="version-name">${model.version}${model.version === DEFAULT_VERSION ? '<span class="badge alt">いちばんkobajinに近い</span>' : ""}${cached.has(model.version) ? '<span class="badge you">保存済み</span>' : ""}</span>
        <span class="version-note">${escapeHtml(model.note || model.summary)}</span>
      </span>
    </label>`).join("");

  const button = $("model-start");
  const status = $("model-status");
  $("model-progress").hidden = state !== "loading";
  $("model-progress-bar").style.width = `${Math.round(progress.ratio * 100)}%`;
  button.disabled = state === "unsupported";
  button.dataset.action = state === "online" || state === "loading" ? "stop" : "start";
  button.textContent = {
    online: "停止してモックに戻す",
    loading: "読み込みを中止",
    error: "もう一度試す",
    unsupported: "このブラウザでは使えません",
    elsewhere: `${chosen} をこのタブで起動`,
  }[state] || (cached.has(chosen) ? `${chosen} を起動` : `${chosen} をダウンロードして起動`);
  status.textContent = {
    online: `${chosen} がこの端末の中で投稿・返信・リアクションを書いています。`,
    loading: progress.text,
    error: errorText,
    elsewhere: "kobajinAIは一度に1つのタブでだけ動きます（2つ動かすとメモリとGPUを2倍使うため）。起動すると、もう一方のタブはモックに戻ります。",
    unsupported: "WebGPUに対応したブラウザ（最新のChromeやEdgeなど）で開くと、kobajinAIを動かせます。それまではモック推論（定型文）で動きます。",
  }[state] || (cached.has(chosen)
    ? "このブラウザに保存済みなので、すぐに起動できます。"
    : "初回だけHugging Faceからモデル（約2.2GB）をダウンロードして、このブラウザに保存します。Wi-Fiでの利用をおすすめします。");
}

function goOnline(engine, version) {
  llm.use(createWebLLM(engine, version, (error) => {
    // The engine broke (GPU lost, tab out of memory): back to the mock.
    // Not online means it was being unloaded on purpose (stop, switch,
    // another tab), which cuts a running generation short.
    if (state !== "online") return;
    console.error("kobajinAI generation failed", error);
    llm.use(null);
    unload();
    state = "error";
    errorText = "生成中にエラーが起きたため、モック推論に戻しました。";
    render();
    refreshAll(posts);
  }));
  state = "online";
  cached.add(version);
  render();
  refreshAll(posts);
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
// finds the model running elsewhere stays on the mock.
async function start({ steal = true } = {}) {
  if (!HAS_WEBGPU) return;
  const version = chosen;
  llm.use(null);
  state = "loading";
  progress = { ratio: 0, text: "準備しています…" };
  write(STARTED_KEY, "1");
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
    errorText = `${error?.message || "読み込みに失敗しました"}。通信とブラウザの空き容量を確かめて、もう一度試してください。`;
    render();
  }
}

async function stop() {
  write(STARTED_KEY, null);
  llm.use(null);
  state = "idle";
  render();
  refreshAll(posts);
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

// Called when the visitor posts or replies.
export function wakeModel() {
  if (state === "idle" && read(STARTED_KEY) && cached.has(chosen)) start({ steal: false });
}

// Another tab started the model and took it from this one.
function evicted() {
  if (state !== "online" && state !== "loading") return;
  llm.use(null);
  state = "elsewhere";
  render();
  refreshAll(posts);
  toast("別のタブでkobajinAIを起動したので、このタブはモックに戻りました");
}

export function setupModelPicker() {
  onEvict(evicted);
  const dialog = $("model-dialog");
  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-open-model]")) { render(); dialog.showModal(); }
  });
  dialog.addEventListener("click", (event) => {
    if (event.target.closest("[data-close]") || event.target === dialog) dialog.close();
  });
  $("model-versions").addEventListener("change", (event) => {
    if (event.target.name !== "model-version") return;
    chosen = event.target.value;
    write(MODEL_KEY, chosen);
    // Switching while running loads the new version in its place.
    if (state === "online" || state === "loading") start(); else { if (state === "error") state = "idle"; render(); }
  });
  $("model-start").addEventListener("click", () => {
    if ($("model-start").dataset.action === "stop") stop(); else start();
  });
  render();
  // Marks saved versions, once the page has settled; nothing is loaded
  // until the visitor starts one.
  if (HAS_WEBGPU) {
    setTimeout(() => {
      Promise.all(MODELS.map(async (model) => { if (await isCached(model.version)) cached.add(model.version); }))
        .then(() => { if (state !== "online" && state !== "loading") render(); })
        .catch(() => {});
    }, 1500);
  }
}
