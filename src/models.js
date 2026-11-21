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

export const MODELS = [
  {
    version: "v1",
    id: "kobajinAI-qwen35-4b-kobajin-clean-v1-8d6dab1-q4f16_1-MLC",
    url: `${REPO}/8d6dab15195505c8c92c2d9ceb99f2551239f2d4/v1/`,
    summary: "Real chats only. Casual voice; often ignores memories.",
    note: "実際の会話だけで学習。くだけた口調だが、記憶はあまり使わない。",
  },
  {
    version: "v2",
    id: "kobajinAI-qwen35-4b-kobajin-clean-v2-1f9287d-q4f16_1-MLC",
    url: `${REPO}/1f9287d837eb311f5d0ab9262240ab2ed362d87c/v2/`,
    summary: "Real chats, plus practice using memories.",
    note: "実際の会話に加えて、記憶の使い方を練習。関係ない記憶を混ぜることもある。",
  },
  {
    version: "v3",
    id: "kobajinAI-qwen35-4b-luna-v3-a2534a2-q4f16_1-MLC",
    url: `${REPO}/a2534a20253d54cce925307fbd76086498b09a11/`,
    summary: "Mostly Luna-written examples. Longer, advice-like replies.",
    note: "Lunaが書いた例が中心。長めでアドバイス寄りの返事になりやすい。",
  },
];

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
