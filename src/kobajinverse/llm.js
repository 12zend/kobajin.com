import { createMockLLM } from "./mock_llm.js";
import { PROMPTS } from "./prompts.js";

// Inference layer
//
// Everything the page asks of kobajinAI goes through one call:
//
//     llm.complete(task, input) -> Promise<string>   (raw model text)
//
// task is one of "post" | "reply" | "elaborate" | "react" | "judge" |
// "trend". Nothing answers until a kobajinAI version is running
// (model_picker.js, engine.js): until then, and whenever it stops, a call
// waits for the next one. The site never fills in with canned text, so a
// screenshot of the page always shows lines the model wrote (AGENTS.md).
// llm.use() swaps in createWebLLM(), or null to stop. In development,
// `?mock` uses the mock (mock_llm.js), which ignores PROMPTS. Cleaning,
// length limits, gates, and parsing live in brain.js, so both return raw
// text.
//
// llm.source names what wrote a line: the version ("v2") or "mock".

// `onFail` hears about an engine error (the call still rejects).
export function createWebLLM(engine, version, onFail = () => {}) {
  return {
    name: "webllm",
    version,
    async complete(task, input) {
      const prompt = PROMPTS[task];
      if (!prompt) throw new Error(`unknown task: ${task}`);
      try {
        const reply = await engine.chat.completions.create({
          messages: prompt.messages(input),
          ...prompt.decoding,
          extra_body: { enable_thinking: false },
        });
        return reply.choices[0]?.message?.content || "";
      } catch (error) {
        onFail(error);
        throw error;
      }
    },
  };
}

export const mockLLM = () => createMockLLM();

let active = null;
let wake = () => {};
let ready = new Promise((resolve) => { wake = resolve; });

export const llm = {
  get name() { return active?.name || "none"; },
  get source() { return !active ? "" : active.name === "mock" ? "mock" : active.version; },
  async complete(task, input) {
    while (!active) await ready;
    return active.complete(task, input);
  },
  use(next) {
    active = next || null;
    if (active) wake();
    else ready = new Promise((resolve) => { wake = resolve; });
  },
};
