import { MODELS, findModel } from "../models.js";

// WebLLM engine lifecycle: one version at a time, in a worker, in one tab.
//
// AGENTS.md: the site downloads and caches only the version a visitor
// picks, and keeps only that one in memory. load() unloads the previous
// version first, and stops a version that is still loading, so two
// versions are never in memory together. A browser-wide lock (Web Locks)
// keeps it to one tab too: a tab that starts the model takes it from any
// other tab, which goes back to the mock (onEvict). WebLLM itself is
// imported only here, on first use, so a visitor who never starts the
// model never downloads it.

const DEV_FLAGS = import.meta.env.DEV ? new URLSearchParams(location.search) : new URLSearchParams();
// `?no-webgpu` (dev only) previews the unsupported-browser state.
export const HAS_WEBGPU = Boolean(navigator.gpu) && !DEV_FLAGS.has("no-webgpu");

// The timeline's prompts are short: the system prompt, at most three
// memories, a few log lines, and up to 120 new tokens, well under 1,500
// tokens in all. A 2,048-token window instead of the bundles' 4,096
// halves the KV cache the GPU keeps. The parity page keeps the bundles'
// own window (chat_request.js).
const CONTEXT_WINDOW = 2048;

let webllm = null;
async function lib() {
  if (!webllm) {
    const [mlc, request] = await Promise.all([import("@mlc-ai/web-llm"), import("../chat_request.js")]);
    const appConfig = request.buildAppConfig();
    if (!appConfig) throw new Error(`このバージョンのWebLLMに ${request.BASE_MODEL_ID} がありません`);
    const ids = new Set(MODELS.map((model) => model.id));
    appConfig.model_list = appConfig.model_list.map((item) => (ids.has(item.model_id)
      ? { ...item, overrides: { ...item.overrides, context_window_size: CONTEXT_WINDOW } }
      : item));
    webllm = { mlc, appConfig };
  }
  return webllm;
}

export async function isCached(version) {
  if (!HAS_WEBGPU) return false;
  try {
    const { mlc, appConfig } = await lib();
    return Boolean(await mlc.hasModelInCache(findModel(version).id, appConfig));
  } catch {
    return false;
  }
}

// Another tab holds the model and this start was not allowed to take it.
export class EngineBusyError extends Error {
  constructor() { super("別のタブでkobajinAIが動いています"); this.name = "EngineBusyError"; }
}

const LOCK_NAME = "kobajinverse.engine";
let lockSeq = 0;
let held = null; // { id, free }: this tab's hold on the lock
let evictListener = () => {};

// Called when another tab takes the model from this one.
export function onEvict(listener) { evictListener = listener; }

// Resolves to { id, free } once this tab holds the lock, or null when
// `steal` is false and another tab holds it.
function acquireLock(steal) {
  if (!navigator.locks) return Promise.resolve({ id: 0, free: () => {} });
  const id = ++lockSeq;
  return new Promise((resolve) => {
    navigator.locks.request(LOCK_NAME, steal ? { steal: true } : { ifAvailable: true }, (lock) => {
      if (!lock) { resolve(null); return undefined; }
      return new Promise((free) => resolve({ id, free }));
    }).catch((error) => {
      // Stolen by another tab: free this tab's copy.
      if (error?.name === "AbortError" && held?.id === id) evicted();
    });
  });
}

function releaseLock() {
  const was = held;
  held = null;
  was?.free();
}

// The listener moves the page to the mock first, so a generation the
// unload cuts short is not taken for a broken engine.
async function evicted() {
  held = null;
  if (current || pending) evictListener();
  await unload();
}

let current = null; // { version, engine, worker }
let pending = null; // { cancel }: a version still loading
let loadToken = 0;

export const loadedVersion = () => current?.version || null;

export async function unload() {
  loadToken += 1;
  const loading = pending;
  pending = null;
  loading?.cancel();
  const was = current;
  current = null;
  if (was) {
    try { await was.engine.unload(); } catch { /* terminating frees it either way */ }
    was.worker.terminate();
  }
  releaseLock();
}

// Resolves to the engine, or null when another load() or unload()
// superseded this one. Rejects when loading fails, or with
// EngineBusyError when `steal` is false and another tab has the model.
export async function load(version, onProgress = () => {}, { steal = true } = {}) {
  await unload();
  const token = ++loadToken;
  const { mlc, appConfig } = await lib();
  if (token !== loadToken) return null;
  const lock = await acquireLock(steal);
  if (!lock) throw new EngineBusyError();
  if (token !== loadToken) { lock.free(); return null; }
  held = lock;

  const model = findModel(version);
  const worker = new Worker(new URL("../webllm.worker.js", import.meta.url), { type: "module" });
  // A superseded load stops at once instead of finishing in the
  // background next to the new one.
  let abort;
  const cancelled = new Promise((_, reject) => { abort = reject; });
  const mine = { cancel: () => { worker.terminate(); abort(new Error("cancelled")); } };
  pending = mine;
  try {
    const engine = await Promise.race([
      mlc.CreateWebWorkerMLCEngine(worker, model.id, {
        appConfig,
        initProgressCallback: (report) => { if (token === loadToken) onProgress(report); },
      }),
      cancelled,
    ]);
    if (pending === mine) pending = null;
    if (token !== loadToken) {
      engine.unload().catch(() => {}).finally(() => worker.terminate());
      return null;
    }
    current = { version: model.version, engine, worker };
    return { engine, version: model.version };
  } catch (error) {
    worker.terminate();
    if (token !== loadToken) return null;
    if (pending === mine) pending = null;
    releaseLock();
    throw error;
  }
}
