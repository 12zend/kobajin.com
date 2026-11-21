import { createMockLLM } from "./mock_llm.js";
import { PROMPTS } from "./prompts.js";

// Inference layer
//
// Everything the page asks of kobajinAI goes through one call:
//
//     llm.complete(task, input) -> Promise<string>   (raw model text)
//
// task is one of "post" | "reply" | "elaborate" | "react" | "judge" |
// "trend". The page starts on the mock (mock_llm.js), which ignores
// PROMPTS and returns canned text. When the visitor starts a kobajinAI
// version (model_picker.js, engine.js), llm.use() swaps in createWebLLM()
// and back to the mock when it stops. Cleaning, length limits, gates, and
// parsing live in brain.js, so both return raw text.
//
// llm.source names what wrote a line: "mock" or the version ("v2").

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

const mock = createMockLLM();
let active = mock;

export const llm = {
  get name() { return active.name; },
  get source() { return active.name === "mock" ? "mock" : active.version; },
  complete: (task, input) => active.complete(task, input),
  use(next) { active = next || mock; },
};
