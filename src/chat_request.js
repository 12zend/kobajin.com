// How the site asks WebLLM for a chat reply. The site and the WebLLM parity
// page (self_twin/webllm/parity/) both use this module, so a parity run
// measures the same prompt and decoding visitors get.

import { prebuiltAppConfig } from "@mlc-ai/web-llm";
import { getMemoryContext } from "./memory.js";
import { MODELS, familyOf } from "./models.js";
import SYSTEM_PROMPT_SOURCE from "../self_twin/inference/system_prompt.txt?raw";

export const SYSTEM_PROMPT = SYSTEM_PROMPT_SOURCE.trim();

// The published bundles' mlc-chat-config.json (MLC's `qwen3_5` template) puts
// "<think>" in the assistant role. With enable_thinking: false, WebLLM then
// appends its own "<think>\n\n</think>\n\n", so the prompt ended in
// "assistant\n<think>\n<think>\n\n</think>\n\n", a shape the model never saw in
// training. With the bare role, the prompt is token-for-token the one
// Transformers builds with enable_thinking=False (the training and evaluation
// prompt). Overriding it here leaves the published bundle files unchanged.
// LFM2.5 bundles keep their own template (MLC's `lfm2`), which already matches
// the chat template they were trained with.
export const QWEN_CONV_CONFIG = {
  roles: { user: "<|im_start|>user", assistant: "<|im_start|>assistant" },
  stop_token_ids: [248046, 248044],
};

// Decoding for chat replies. Evaluation uses the same values; see
// self_twin/inference/generation.py for how Transformers matches WebLLM's
// repetition penalty, which counts only generated tokens.
export const CHAT_DECODING = {
  temperature: 0.72,
  top_p: 1,
  repetition_penalty: 1.12,
  max_tokens: 160,
};

const CONV_CONFIGS = { "qwen3.5-4b": QWEN_CONV_CONFIG, "qwen3.5-2b": QWEN_CONV_CONFIG };

// WebLLM's prebuilt record for a version's base model (library, settings).
export function libraryRecord(model) {
  const library = familyOf(model).library;
  const record = prebuiltAppConfig.model_list.find((item) => item.model_id === library);
  if (!record) throw new Error(`このバージョンのWebLLMに ${library} がありません`);
  return record;
}

// The bundle's own overrides plus, for Qwen3.5, the conv template fix above.
export function overridesFor(model) {
  const base = libraryRecord(model);
  const conv = CONV_CONFIGS[model.family];
  return conv ? { ...base.overrides, conv_config: { ...base.overrides?.conv_config, ...conv } } : { ...base.overrides };
}

// WebLLM's prebuilt models plus every kobajinAI version, each on its base
// model's prebuilt library.
export function buildAppConfig() {
  return {
    ...prebuiltAppConfig,
    model_list: [
      ...prebuiltAppConfig.model_list,
      ...MODELS.map((model) => ({ ...libraryRecord(model), model_id: model.id, model: model.url, overrides: overridesFor(model) })),
    ],
  };
}

// Extra request fields a version needs (see FAMILIES in models.js).
export const extraBodyFor = (model) => familyOf(model).extraBody;

export function buildSystemPrompt(query) {
  const memoryContext = getMemoryContext(query);
  return memoryContext ? `${SYSTEM_PROMPT}\n\n${memoryContext}` : SYSTEM_PROMPT;
}

export function formatConversationLog(messages) {
  return messages.slice(-12).map(({ role, content, timestamp }) => {
    const date = new Date(timestamp || Date.now());
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hour = String(date.getHours()).padStart(2, "0");
    const minute = String(date.getMinutes()).padStart(2, "0");
    const speaker = role === "assistant" ? "自分" : "参加者A";
    return `${month}-${day} ${hour}:${minute} 【${speaker}】${content}`;
  }).join("\n");
}
