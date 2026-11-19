#!/usr/bin/env node
// Line-delimited JSON bridge so Python evaluation uses the browser's own gate,
// token-budget, and public-memory code (src/gates.js, src/memory.js) instead of
// a re-implementation. One request per stdin line: {"op": "...", ...args}.
// One response per stdout line: {"ok": true, "result": ...} or {"ok": false, "error": "..."}.

import { registerHooks } from "node:module";
import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

// Vite resolves JSON imports and `?raw` text imports; give Node the same view.
registerHooks({
  resolve(specifier, context, nextResolve) {
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    const plain = url.split("?")[0];
    if (url.includes("?raw")) {
      const text = readFileSync(fileURLToPath(plain), "utf8");
      return { format: "module", source: `export default ${JSON.stringify(text)};`, shortCircuit: true };
    }
    if (plain.endsWith(".json")) {
      const text = readFileSync(fileURLToPath(plain), "utf8");
      return { format: "module", source: `export default ${text};`, shortCircuit: true };
    }
    return nextLoad(url, context);
  },
});

const gates = await import("../../src/gates.js");
const memory = await import("../../src/memory.js");

const ops = {
  estimate_tokens: ({ text }) => gates.estimateTokens(text),
  trim_to_budget: ({ text, max_tokens }) => gates.trimToTokenBudget(text, max_tokens),
  clean: ({ text }) => gates.cleanModelText(text),
  parse_json: ({ text }) => gates.parseJsonOutput(text),
  form: ({ text, finish_reason }) => gates.formGate(text, finish_reason ?? null),
  fact_rule: ({ text, excerpt, title }) => gates.factRuleGate(text, excerpt, title),
  filter_facts: ({ facts, excerpt, title }) => gates.filterSupportedFacts(facts, excerpt, title),
  identity: ({ text }) => gates.identityGate(text, memory.findMemoryRecords(text)),
  novelty: ({ text, recent, excerpt }) => {
    const result = gates.noveltyGate(text, recent || [], excerpt || "");
    return { ...result, maxJaccard: Number(result.maxJaccard.toFixed(3)) };
  },
  memory_context: ({ query }) => memory.getMemoryContext(query || ""),
};

const lines = createInterface({ input: process.stdin });
for await (const line of lines) {
  if (!line.trim()) continue;
  try {
    const request = JSON.parse(line);
    const op = ops[request.op];
    if (!op) throw new Error(`Unknown op: ${request.op}`);
    process.stdout.write(`${JSON.stringify({ ok: true, result: op(request) })}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ ok: false, error: String(error?.message || error) })}\n`);
  }
}
