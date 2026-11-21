import { createMockLLM } from "./mock_llm.js";

// Inference layer
//
// Everything the page asks of kobajinAI goes through one call:
//
//     llm.complete(task, input) -> Promise<string>   (raw model text)
//
// task is one of "post" | "reply" | "elaborate" | "react" | "judge" |
// "trend". PROMPTS (prompts.js) say what each task asks; createMockLLM()
// (mock_llm.js) ignores them and returns canned text. To switch to WebLLM,
// write createWebLLM(engine) with the same complete(): send
// PROMPTS[task].system plus PROMPTS[task].user(input) to
// engine.chat.completions.create with PROMPTS[task].decoding and
// extra_body: { enable_thinking: false }, and return
// choices[0].message.content. Then export it here instead of the mock.
// Cleaning, length limits, and parsing live in brain.js, so the swap only
// returns raw text.

export const llm = createMockLLM();
