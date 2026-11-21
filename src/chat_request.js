// How the site asks WebLLM for a chat reply. The site and the WebLLM parity
// page (self_twin/webllm/parity/) both use this module, so a parity run
// measures the same prompt and decoding visitors get.

import { prebuiltAppConfig } from "@mlc-ai/web-llm";
import { getMemoryContext } from "./memory.js";
import { MODELS } from "./models.js";
import SYSTEM_PROMPT_SOURCE from "../self_twin/inference/system_prompt.txt?raw";

// Every version is a Qwen3.5-4B q4f16_1 bundle, so all reuse WebLLM's prebuilt library.
export const BASE_MODEL_ID = "Qwen3.5-4B-q4f16_1-MLC";
export const SYSTEM_PROMPT = SYSTEM_PROMPT_SOURCE.trim();

// The published bundles' mlc-chat-config.json (MLC's `qwen3_5` template) puts
// "<think>" in the assistant role. With enable_thinking: false, WebLLM then
// appends its own "<think>\n\n</think>\n\n", so the prompt ended in
// "assistant\n<think>\n<think>\n\n</think>\n\n", a shape the model never saw in
// training. With the bare role, the prompt is token-for-token the one
// Transformers builds with enable_thinking=False (the training and evaluation
// prompt). Overriding it here leaves the published bundle files unchanged.
export const CONV_CONFIG = {
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

export function buildAppConfig() {
  const baseRecord = prebuiltAppConfig.model_list.find((item) => item.model_id === BASE_MODEL_ID);
  if (!baseRecord) return null;
  const overrides = {
    ...baseRecord.overrides,
    conv_config: { ...baseRecord.overrides?.conv_config, ...CONV_CONFIG },
  };
  return {
    ...prebuiltAppConfig,
    model_list: [
      ...prebuiltAppConfig.model_list,
      ...MODELS.map((model) => ({ ...baseRecord, model_id: model.id, model: model.url, overrides })),
    ],
  };
}

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
