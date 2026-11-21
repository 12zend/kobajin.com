import { DEFAULT_VERSION, MODELS, familyOf, findModel } from "../models.js";
import { $, escapeHtml, say, toast } from "./dom.js";
import { DEV_FLAGS, EngineBusyError, HAS_WEBGPU, gpuProblem, isCached, load, onEvict, unload } from "./engine.js";
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
// downloads that version once (about 2.4 GB for a Qwen3.5-4B version, less
// for the smaller ones; FAMILIES in models.js) and keeps it in this
// browser; only the chosen version is downloaded and loaded. The first
// choice is the version closest to kobajin, or the light version on a
// low-memory device (LOW_MEMORY_DEVICE below).
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

// Free browser storage below the chosen version's weights plus this gets a
// warning: WebLLM's library and room to spare.
const SPARE_BYTES = 0.8e9;
const GB = 1e9;

// The lightest published version (FAMILIES in models.js), if any.
const lightest = () => MODELS.filter((model) => familyOf(model).light).at(-1) || null;

// A device that probably cannot run a Qwen3.5-4B version: it reports less
// than 8 GB (navigator.deviceMemory, Chromium only, capped at 8), or it is a
// phone or tablet (Safari reports no memory; iPadOS presents itself as a Mac
// with a touch screen).
const DEVICE_MEMORY = typeof navigator.deviceMemory === "number" ? navigator.deviceMemory : null;
const MOBILE = navigator.userAgentData?.mobile
  ?? (/iPhone|iPad|iPod|Android|Mobile/i.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1));
const LOW_MEMORY_DEVICE = (DEVICE_MEMORY !== null && DEVICE_MEMORY < 8) || MOBILE;

// Owner decision (2026-11-22): a low-memory device starts on the light
// version; others start on the version closest to kobajin. A version the
// visitor picked before (MODEL_KEY) still wins.
const recommended = () => (LOW_MEMORY_DEVICE && lightest()?.version) || DEFAULT_VERSION;
let chosen = findModel(read(MODEL_KEY) || recommended()).version;
// checking | idle | loading | online | error | elsewhere | unsupported
let state = !HAS_WEBGPU ? "unsupported" : read(STARTED_KEY) ? "checking" : "idle";
let progress = { ratio: 0, text: "" };
let errorText = "";
let unsupportedText = "WebGPUに対応したブラウザ（最新のChrome・Edge、Safari 26以降）で開くと、kobajinAIを動かせます。";
let freeBytes = null; // browser storage left, when the browser says
const cached = new Set();

// Resolves the first time a version comes online; the timeline starts then.
let markReady;
export const modelReady = new Promise((resolve) => { markReady = resolve; });

const percent = () => `${Math.round(progress.ratio * 100)}%`;
const size = (bytes) => `約${(bytes / GB).toFixed(1)}GB`;
const family = () => familyOf(findModel(chosen));

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
  }[state] || (cached.has(chosen) ? `${chosen} を起動` : `${chosen} をダウンロードして始める（${size(family().downloadBytes)}）`);
}

function statusText() {
  return {
    checking: "このブラウザに保存済みのモデルを確認しています…",
    online: `${chosen} がこの端末の中で投稿・返信・リアクションを書いています。`,
    loading: progress.text,
    error: errorText,
    elsewhere: "kobajinAIは一度に1つのタブでだけ動きます（2つ動かすとメモリとGPUを2倍使うため）。起動すると、もう一方のタブは止まります。",
    unsupported: unsupportedText,
  }[state] || (cached.has(chosen)
    ? "このブラウザに保存済みなので、ダウンロードせずに起動できます。"
    : "Hugging Faceからモデルをダウンロードして、このブラウザに保存します。次からはダウンロードなしで起動します。");
}

// What each size of version asks of the device, for the start screen and
// the recommended-specs list. The light (LFM2.5) numbers are WebLLM's
// estimate; they are not yet measured on a 4 GB device.
function needs(fam) {
  if (fam.light) {
    return {
      memory: `GPUが使えるメモリが約${(fam.gpuMB / 1000).toFixed(1)}GB。メモリ4GBのタブレットやスマホでも動く見込みです`,
      note: "軽量版です。WebGPUとshader-f16に対応した端末なら、スマホやタブレットでも動く見込みです。古い端末では遅くなることがあります。",
      warn: "軽量版です。メモリの少ない端末では、ほかのアプリを閉じてから始めてください。",
      browser: "WebGPUに対応したブラウザ（最新のChrome・Edge、Safari 26以降）。スマホやタブレットでも、WebGPUに対応していれば動く見込みです。",
    };
  }
  if (fam.gpuMB < 3000) {
    return {
      memory: `GPUが使えるメモリが約${(fam.gpuMB / 1000).toFixed(1)}GB。メモリ8GBのMacやPCでも動く見込みです`,
      note: "中くらいの大きさの版です。メモリ8GBのMacやPCでも動く見込みですが、ほかのタブを閉じると安定します。スマホでは重いことがあります。",
      warn: "メモリ8GBの端末では、ほかのタブやアプリを閉じてから始めてください。",
      browser: "WebGPUに対応したパソコンのブラウザ（最新のChrome・Edge、Safari 26以降）。スマホでは重いことがあります。",
    };
  }
  return {
    memory: "GPUが使えるメモリが約4GB。メモリ16GB以上のApple Silicon Mac、またはVRAM 6GB以上のGPUを積んだPCなら快適です",
    note: "メモリ8GBのMacや内蔵GPUのPCでも動くことはありますが、遅くなったり、ほかのタブを閉じる必要があったりします。スマホではほとんど動きません。",
    warn: "メモリ8GBの端末では、重くなったり途中で止まったりすることがあります。ほかのタブやアプリを閉じてから始めてください。",
    browser: "WebGPUに対応したパソコンのブラウザ（最新のChrome・Edge、Safari 26以降）。スマホではほとんど動きません。",
  };
}

// Storage and memory notes on the start screen. The memory note is always
// there; a low-memory device (LOW_MEMORY_DEVICE) gets a stronger one when
// the chosen version is not a light one, pointing to the light version if
// one is published.
function renderNotes() {
  const fam = family();
  const need = needs(fam);
  const lowMemory = LOW_MEMORY_DEVICE && !fam.light;
  const light = lightest();
  const device = DEVICE_MEMORY !== null && DEVICE_MEMORY < 8 ? `この端末のメモリは${DEVICE_MEMORY}GB程度のようです。` : "スマホやタブレットでは、";
  $("gate-memory").textContent = !lowMemory
    ? need.warn
    : light
      ? `${device}${chosen} はおそらく動きません。軽量版の ${light.version} を選んでください。`
      : `${device}kobajinAIはおそらく動きません。`;
  $("gate-memory").closest("li").dataset.warn = String(lowMemory);
  const download = size(fam.downloadBytes);
  const lowStorage = freeBytes !== null && freeBytes < fam.downloadBytes + SPARE_BYTES && !cached.has(chosen);
  $("gate-storage").textContent = lowStorage
    ? `このブラウザで使える空きが${size(freeBytes)}しかありません。${size(fam.downloadBytes + SPARE_BYTES)}以上の空きを作ってから始めてください。`
    : `端末のストレージに空きが必要です（モデルは${download}）。`;
  $("gate-storage").closest("li").dataset.warn = String(lowStorage);
  $("gate-browser").textContent = need.browser;
  $("gate-network").textContent = `はじめの1回だけ${download}を受信します。Wi-Fiでの利用をおすすめします。`;
  $("spec-title").textContent = `推奨スペック（${chosen}）`;
  $("spec-memory").textContent = need.memory;
  $("spec-storage").textContent = `ストレージに${size(fam.downloadBytes + SPARE_BYTES)}程度（モデルは${download}）`;
  $("spec-note").textContent = need.note;
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
        <span class="version-name">${model.version}${model.version === DEFAULT_VERSION ? '<span class="badge alt">いちばんkobajinに近い</span>' : ""}${familyOf(model).light ? '<span class="badge you">軽量</span>' : ""}${LOW_MEMORY_DEVICE && model.version === recommended() ? '<span class="badge alt">この端末におすすめ</span>' : ""}${cached.has(model.version) ? '<span class="badge you">保存済み</span>' : ""}</span>
        <span class="version-note">${escapeHtml(model.note || model.summary)}</span>
      </span>
    </label>`).join("");
  $("gate-version").innerHTML = MODELS.map((model) => `<option value="${model.version}"${model.version === chosen ? " selected" : ""}>${model.version}${model.version === DEFAULT_VERSION ? "（いちばんkobajinに近い）" : ""}${familyOf(model).light ? "・軽量" : ""}${LOW_MEMORY_DEVICE && model.version === recommended() ? "・この端末におすすめ" : ""}${cached.has(model.version) ? "・保存済み" : ""}</option>`).join("");
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
  // Every version needs shader-f16 (q4f16_1); without it WebLLM fails after
  // the download, so say so up front.
  const problem = await gpuProblem();
  if (problem) {
    unsupportedText = problem;
    state = "unsupported";
    render();
    return;
  }
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
