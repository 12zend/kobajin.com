// Runs fixed prompts through the real WebLLM engine with the site's prompt
// building, then posts the run to the dev server, which saves it under
// self_twin/evaluation/webllm_parity/. self_twin/evaluation/webllm_parity.py
// compares a run with Transformers. Dev only (npm run dev, then open
// /self_twin/webllm/parity/).
//
//   version   published version to load (default: the site's default)
//   template  site   = the site's conv template override (src/chat_request.js)
//             bundle = the bundle's own mlc-chat-config.json template
//   mode      greedy = top-1 decoding with logprobs, repetition penalty off
//             sampled = the site's CHAT_DECODING, `samples` replies per prompt
//             forced = for each position in `ref` (from
//             self_twin/evaluation/webllm_forced.py reference), WebLLM's most
//             likely next token after that exact prefix, via text completion
//   ref       reference positions for forced mode (file name without .json in
//             self_twin/evaluation/webllm_parity/)
//   bundle_url  optional local bundle URL to load instead of the published
//             one, e.g. http://127.0.0.1:8001/resolve/local/ from
//             self_twin/webllm/serve_local_bundle.py --bundle kobajin-clean-v2
//   cache     WebLLM cache backend: cache (the site's default), indexeddb, or
//             memory (keeps the weights in page memory only, for browsers
//             whose site storage can't hold a 2.2 GB bundle)
//   name      file name for the saved run

import { CreateMLCEngine, prebuiltAppConfig } from "@mlc-ai/web-llm";
import { BASE_MODEL_ID, CHAT_DECODING, buildAppConfig, buildSystemPrompt, formatConversationLog } from "../../../src/chat_request.js";
import { DEFAULT_VERSION, findModel } from "../../../src/models.js";
import PROMPTS from "./prompts.json";
import WEBLLM_PACKAGE from "../../../node_modules/@mlc-ai/web-llm/package.json";

const params = new URLSearchParams(location.search);
const model = findModel(params.get("version") || DEFAULT_VERSION);
const template = params.get("template") === "bundle" ? "bundle" : "site";
const mode = ["sampled", "forced"].includes(params.get("mode")) ? params.get("mode") : "greedy";
const samples = Number(params.get("samples") || 8);
const maxTokens = Number(params.get("max_tokens") || (mode === "greedy" ? 64 : CHAT_DECODING.max_tokens));
const localBundle = params.get("bundle_url");
const modelUrl = localBundle || model.url;
const name = params.get("name") || `${model.version}-${template}-${mode}`;
// A fixed local time so the "MM-DD HH:MM" prefix is the same in every run.
const TIMESTAMP = new Date(2026, 10, 21, 20, 0).getTime();

const statusLine = document.querySelector("#status");
const out = document.querySelector("#out");
const status = (text) => { statusLine.textContent = text; };
const log = (text) => { out.textContent += `${text}\n`; };

// A Cache Storage stand-in that keeps responses in memory for this page only.
function useMemoryCaches() {
  const stores = new Map();
  const openStore = (name) => {
    if (!stores.has(name)) stores.set(name, new Map());
    const entries = stores.get(name);
    const key = (request) => (typeof request === "string" ? request : request.url);
    const keep = async (request, response) => {
      entries.set(key(request), { body: await response.arrayBuffer(), status: response.status, headers: [...response.headers] });
    };
    return {
      async add(request) {
        const response = await fetch(request);
        if (!response.ok) throw new TypeError(`${response.status} for ${key(request)}`);
        await keep(request, response);
      },
      put: keep,
      async match(request) {
        const entry = entries.get(key(request));
        return entry ? new Response(entry.body, { status: entry.status, headers: entry.headers }) : undefined;
      },
      async delete(request) { return entries.delete(key(request)); },
      async keys() { return [...entries.keys()].map((url) => new Request(url)); },
    };
  };
  const shim = {
    open: async (name) => openStore(name),
    has: async (name) => stores.has(name),
    delete: async (name) => stores.delete(name),
    keys: async () => [...stores.keys()],
    match: async () => undefined,
  };
  Object.defineProperty(window, "caches", { value: shim, configurable: true });
}

function appConfigFor() {
  const appConfig = buildAppConfig();
  appConfig.model_list.find((item) => item.model_id === model.id).model = modelUrl;
  if (params.get("cache") === "memory") useMemoryCaches();
  else if (params.get("cache")) appConfig.cacheBackend = params.get("cache");
  if (template === "bundle") {
    const baseRecord = prebuiltAppConfig.model_list.find((item) => item.model_id === BASE_MODEL_ID);
    const record = appConfig.model_list.find((item) => item.model_id === model.id);
    record.overrides = baseRecord.overrides;
  }
  return appConfig;
}

// WebLLM's RNG is a Park–Miller LCG: nearby seeds (1000, 1001, …) give nearly
// the same first random number, so every sample would pick the same first token.
function spreadSeed(sample) {
  return Number((BigInt(sample + 1) * 2654435761n) % 2147483647n);
}

function messagesFor(prompt) {
  return [
    { role: "system", content: buildSystemPrompt(prompt) },
    { role: "user", content: formatConversationLog([{ role: "user", content: prompt, timestamp: TIMESTAMP }]) },
  ];
}

async function run() {
  status(`Loading ${model.version} (${template} template)…`);
  const engine = await CreateMLCEngine(model.id, {
    appConfig: appConfigFor(),
    initProgressCallback: (report) => status(report.text),
  });
  const pipeline = engine.loadedModelIdToPipeline?.get(model.id);
  const results = [];
  if (mode === "forced") {
    const ref = await (await fetch(`/self_twin/evaluation/webllm_parity/${params.get("ref")}.json`)).json();
    for (const [index, position] of ref.positions.entries()) {
      if (index % 10 === 0) status(`${index}/${ref.positions.length} positions`);
      const reply = await engine.completions.create({
        prompt: position.prefix,
        temperature: 1,
        top_p: 1e-5,
        repetition_penalty: 1,
        max_tokens: 1,
        logprobs: true,
        top_logprobs: 5,
        ignore_eos: true,
      });
      const logprob = reply.choices[0].logprobs?.content?.[0];
      results.push({ token_id: pipeline.outputIds.at(-1), logprob: logprob?.logprob ?? null, top: logprob?.top_logprobs || [] });
    }
  }
  for (const [index, prompt] of (mode === "forced" ? [] : PROMPTS).entries()) {
    status(`${index + 1}/${PROMPTS.length}: ${prompt}`);
    const messages = messagesFor(prompt);
    const item = { prompt, messages, replies: [] };
    const runs = mode === "greedy" ? 1 : samples;
    for (let sample = 0; sample < runs; sample += 1) {
      const request = mode === "greedy"
        // top_p at WebLLM's floor keeps only the most likely token; temperature 1
        // keeps the logprobs equal to the model's own distribution.
        ? { temperature: 1, top_p: 1e-5, repetition_penalty: 1, max_tokens: maxTokens, logprobs: true, top_logprobs: 5 }
        : { ...CHAT_DECODING, max_tokens: maxTokens, seed: spreadSeed(sample) };
      const reply = await engine.chat.completions.create({
        messages,
        ...request,
        extra_body: { enable_thinking: false },
      });
      const choice = reply.choices[0];
      const entry = { text: choice.message.content, finish_reason: choice.finish_reason };
      if (mode === "greedy") {
        entry.logprobs = choice.logprobs?.content || [];
        if (pipeline) {
          entry.output_ids = [...pipeline.outputIds];
          entry.prompt_array = pipeline.conversation.getPromptArray(pipeline.config);
        }
      }
      item.replies.push(entry);
      log(`${prompt} → ${JSON.stringify(entry.text)}`);
    }
    results.push(item);
  }
  const payload = {
    name,
    version: model.version,
    model_id: model.id,
    model_url: modelUrl,
    local_bundle: localBundle || null,
    template,
    mode,
    request: mode === "sampled" ? { ...CHAT_DECODING, max_tokens: maxTokens, samples } : { temperature: 1, top_p: 1e-5, repetition_penalty: 1, max_tokens: mode === "forced" ? 1 : maxTokens },
    ref: mode === "forced" ? params.get("ref") : null,
    webllm_version: WEBLLM_PACKAGE.version,
    gpu: navigator.gpu ? (await navigator.gpu.requestAdapter())?.info?.architecture || null : null,
    created_at: new Date().toISOString(),
    results,
  };
  const saved = await fetch(`/__parity/results?name=${encodeURIComponent(name)}`, { method: "POST", body: JSON.stringify(payload) });
  status(saved.ok ? `Done. Saved ${await saved.text()}` : `Done, but saving failed: ${await saved.text()}`);
  window.__parityDone = true;
  await engine.unload();
}

run().catch((error) => {
  status(`Failed: ${error?.message || error}`);
  console.error(error);
  window.__parityDone = true;
});
