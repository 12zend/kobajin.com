import { findModel } from "../models.js";

// WebLLM engine lifecycle: one version at a time, in a worker.
//
// AGENTS.md: the site downloads and caches only the version a visitor
// picks, and keeps only that one in memory. load() unloads the previous
// version first. WebLLM itself is imported only here, on first use, so a
// visitor who never starts the model never downloads it.

const DEV_FLAGS = import.meta.env.DEV ? new URLSearchParams(location.search) : new URLSearchParams();
// `?no-webgpu` (dev only) previews the unsupported-browser state.
export const HAS_WEBGPU = Boolean(navigator.gpu) && !DEV_FLAGS.has("no-webgpu");

let webllm = null;
async function lib() {
  if (!webllm) {
    const [mlc, request] = await Promise.all([import("@mlc-ai/web-llm"), import("../chat_request.js")]);
    webllm = { mlc, request, appConfig: request.buildAppConfig() };
    if (!webllm.appConfig) throw new Error(`このバージョンのWebLLMに ${request.BASE_MODEL_ID} がありません`);
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

let current = null; // { version, engine, worker }
let loadToken = 0;

export const loadedVersion = () => current?.version || null;

export async function unload() {
  loadToken += 1;
  const was = current;
  current = null;
  if (!was) return;
  try { await was.engine.unload(); } catch { /* terminating frees it either way */ }
  was.worker.terminate();
}

// Resolves to the engine, or null when another load() or unload()
// superseded this one. Rejects when loading fails.
export async function load(version, onProgress = () => {}) {
  await unload();
  const token = ++loadToken;
  const { mlc, appConfig } = await lib();
  if (token !== loadToken) return null;
  const model = findModel(version);
  const worker = new Worker(new URL("../webllm.worker.js", import.meta.url), { type: "module" });
  try {
    const engine = await mlc.CreateWebWorkerMLCEngine(worker, model.id, {
      appConfig,
      initProgressCallback: (report) => { if (token === loadToken) onProgress(report); },
    });
    if (token !== loadToken) {
      engine.unload().catch(() => {}).finally(() => worker.terminate());
      return null;
    }
    current = { version: model.version, engine, worker };
    return { engine, version: model.version };
  } catch (error) {
    worker.terminate();
    throw error;
  }
}
