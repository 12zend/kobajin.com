// Every published kobajinAI version, selectable on the site.
//
// Each version lives in its own folder of kobajin/kobajinAI and is pinned to an
// immutable Hugging Face revision: WebLLM caches weights by URL, so a pinned
// URL never mixes old and new files. To add a version (v4, …), upload its
// bundle to a new folder, then add a record here. Never edit a published
// record's revision or path; publish a new version instead.
// See self_twin/webllm/README.md.

import CLOSEST from "./closest_model.json";

const REPO = "https://huggingface.co/kobajin/kobajinAI/resolve";

// The base model a version was trained from decides which prebuilt WebLLM
// library runs it and how big it is. A fine-tuned bundle has the same tensors
// as MLC's own bundle of its base, so it reuses that library (no .wasm of ours).
//
// - library: WebLLM's prebuilt model_id whose library and settings to reuse.
// - downloadBytes: the q4f16_1 weights (MLC's tensor-cache.json of that base).
// - gpuMB: WebLLM's vram_required_MB for that library (at a 4,096-token window;
//   the site uses 2,048, so a little less).
// - extraBody: sent with every request. Qwen3.5 needs enable_thinking: false
//   for the empty think block it was trained with; WebLLM adds that block to
//   any model it is sent to, so LFM2.5, trained without one, gets nothing.
// - light: small enough for 4 GB tablets and phones (start screen notes).
export const FAMILIES = {
  "qwen3.5-4b": {
    label: "Qwen3.5 4B",
    library: "Qwen3.5-4B-q4f16_1-MLC",
    downloadBytes: 2.367e9,
    gpuMB: 3868,
    extraBody: { enable_thinking: false },
    light: false,
  },
  "qwen3.5-2b": {
    label: "Qwen3.5 2B",
    library: "Qwen3.5-2B-q4f16_1-MLC",
    downloadBytes: 1.059e9,
    gpuMB: 2245,
    extraBody: { enable_thinking: false },
    light: false,
  },
  "lfm2.5-1.2b": {
    label: "LFM2.5 1.2B",
    library: "LFM2.5-1.2B-Instruct-q4f16_1-MLC",
    downloadBytes: 0.659e9,
    gpuMB: 1001,
    extraBody: {},
    light: true,
  },
};

export const MODELS = [
  {
    version: "v1",
    family: "qwen3.5-4b",
    id: "kobajinAI-qwen35-4b-kobajin-clean-v1-8d6dab1-q4f16_1-MLC",
    url: `${REPO}/8d6dab15195505c8c92c2d9ceb99f2551239f2d4/v1/`,
    summary: "Real chats only. Casual voice; often ignores memories.",
    note: "実際の会話だけで学習。くだけた口調だが、記憶はあまり使わない。",
  },
  {
    version: "v2",
    family: "qwen3.5-4b",
    id: "kobajinAI-qwen35-4b-kobajin-clean-v2-1f9287d-q4f16_1-MLC",
    url: `${REPO}/1f9287d837eb311f5d0ab9262240ab2ed362d87c/v2/`,
    summary: "Real chats, plus practice using memories.",
    note: "実際の会話に加えて、記憶の使い方を練習。関係ない記憶を混ぜることもある。",
  },
  {
    version: "v3",
    family: "qwen3.5-4b",
    id: "kobajinAI-qwen35-4b-luna-v3-a2534a2-q4f16_1-MLC",
    url: `${REPO}/a2534a20253d54cce925307fbd76086498b09a11/`,
    summary: "Mostly Luna-written examples. Longer, advice-like replies.",
    note: "Lunaが書いた例が中心。長めでアドバイス寄りの返事になりやすい。",
  },
  {
    version: "v4",
    family: "lfm2.5-1.2b",
    id: "kobajinAI-lfm25-1.2b-jp-kobajin-clean-v2-5c03457-q4f16_1-MLC",
    url: `${REPO}/5c03457fbc811484733abae9e1fa1b55f42e4059/v4/`,
    summary: "Light version: LFM2.5 1.2B trained like v2. Shorter replies.",
    note: "軽量版。v2と同じデータと方法で、小さいLFM2.5-1.2B-JPを学習。返事は短めで、v2より本人らしさは少し下がる。",
  },
];

// Development only: `VITE_LOCAL_MODEL=<family>=<bundle URL>` adds a "local"
// version that loads a bundle served by self_twin/webllm/serve_local_bundle.py,
// e.g. VITE_LOCAL_MODEL=lfm2.5-1.2b=http://127.0.0.1:8001/resolve/local/
if (import.meta.env.DEV && import.meta.env.VITE_LOCAL_MODEL) {
  const [family, url] = import.meta.env.VITE_LOCAL_MODEL.split(/=(.*)/s);
  if (!FAMILIES[family] || !url) throw new Error(`VITE_LOCAL_MODEL needs <family>=<url>, family one of ${Object.keys(FAMILIES)}`);
  MODELS.push({
    version: "local",
    family,
    id: `kobajinAI-local-${family}-dev-q4f16_1-MLC`,
    url,
    summary: "Local development bundle.",
    note: `開発用のローカルバンドル（${FAMILIES[family].label}）。`,
  });
}

// The default is the version currently judged closest to kobajin, named in
// closest_model.json. Update that file (version, date, and the evidence) when
// evaluation finds a closer version or the owner decides; a visitor's own
// choice in the picker still wins. An unknown version falls back to the newest.
export const DEFAULT_VERSION = MODELS.some((model) => model.version === CLOSEST.version)
  ? CLOSEST.version
  : MODELS[MODELS.length - 1].version;

// `VITE_MODEL_URL` points the default version at a local bundle for testing.
if (import.meta.env.VITE_MODEL_URL) {
  const record = MODELS.find((model) => model.version === DEFAULT_VERSION);
  record.url = import.meta.env.VITE_MODEL_URL;
}

export function findModel(version) {
  return MODELS.find((model) => model.version === version) || MODELS.find((model) => model.version === DEFAULT_VERSION);
}

export const familyOf = (model) => FAMILIES[model.family];
