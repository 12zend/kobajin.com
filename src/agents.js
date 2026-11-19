// Role runner and feed pipelines for kobajinAI.
//
// One WebLLM engine plays every role. Each role differs only in its system
// prompt and sampling profile, which live in src/prompts/roles.json so the
// browser and self_twin/evaluation scripts read the same settings. Chat keeps
// using the existing sendMessage path in main.js and is not routed here.

import SYSTEM_PROMPT_SOURCE from "../self_twin/inference/system_prompt.txt?raw";
import ROLE_CONFIG from "./prompts/roles.json";
import seedPrompt from "./prompts/seed.txt?raw";
import tweetPrompt from "./prompts/tweet.txt?raw";
import replyPrompt from "./prompts/reply.txt?raw";
import factCheckPrompt from "./prompts/fact_check.txt?raw";
import { findMemoryRecords, getMemoryContext } from "./memory.js";
import {
  cleanModelText,
  estimateTokens,
  factRuleGate,
  filterSupportedFacts,
  formGate,
  identityGate,
  noveltyGate,
  parseJsonOutput,
  topicOverlap,
  trimToTokenBudget,
} from "./gates.js";

const SYSTEM_PROMPT = SYSTEM_PROMPT_SOURCE.trim();
const PROMPTS = {
  "seed.txt": seedPrompt.trim(),
  "tweet.txt": tweetPrompt.trim(),
  "reply.txt": replyPrompt.trim(),
  "fact_check.txt": factCheckPrompt.trim(),
};
const CONTEXT_WINDOW = ROLE_CONFIG.context_window;
const EXCERPT_TOKEN_BUDGET = ROLE_CONFIG.excerpt_token_budget;
// Chat template tokens, the empty think block, and estimator slack.
const TEMPLATE_MARGIN_TOKENS = ROLE_CONFIG.template_margin_tokens;
const MAX_TWEET_ATTEMPTS = 3; // first try + up to two regenerations
const CHAT_FALLBACK = "ごめんなさい。短い返事を作れませんでした。もう一度お願いします。";

export const ROLES = ROLE_CONFIG.roles;

export class RoleAbortedError extends Error {
  constructor() {
    super("Role generation was interrupted");
    this.name = "RoleAbortedError";
  }
}

// `response_format` is applied per request. If the grammar matcher cannot be
// built for this model, later JSON roles run without it and rely on strict
// JSON.parse, which treats malformed output as a failed check.
let jsonModeAvailable = true;

export function composeSystemPrompt(roleName, context = "") {
  const role = ROLES[roleName];
  if (!role) throw new Error(`Unknown role: ${roleName}`);
  const rolePrompt = PROMPTS[role.prompt];
  const parts = role.compose === "append" ? [SYSTEM_PROMPT, rolePrompt] : [rolePrompt];
  if (context && context.trim()) parts.push(context.trim());
  return parts.join("\n\n");
}

function isContextOverflow(error) {
  const text = `${error?.name || ""} ${error?.message || ""}`;
  return /ContextWindowSizeExceeded|context window size|exceed.*context/i.test(text);
}

function isGrammarError(error) {
  const text = `${error?.name || ""} ${error?.message || ""}`;
  return /GrammarMatcher|grammar|response_format|xgrammar/i.test(text);
}

/**
 * Run one role on the shared engine.
 *
 * input:
 *   user     string user-turn content (ignored when `fit` is given)
 *   context  extra system text appended after the role prompt
 *   fit      { excerpt, apply(excerpt) => { user, context } } – the excerpt is
 *            trimmed first so the request stays inside the context window
 * options:
 *   signal   AbortSignal; aborting interrupts the engine and throws RoleAbortedError
 */
export async function runRole(engine, roleName, input, { signal } = {}) {
  const role = ROLES[roleName];
  if (!role) throw new Error(`Unknown role: ${roleName}`);

  const render = (excerptBudget) => {
    if (!input.fit) {
      return { user: input.user || "", context: input.context || "", excerptTokens: 0 };
    }
    const empty = input.fit.apply("");
    const fixedTokens = estimateTokens(composeSystemPrompt(roleName, empty.context)) + estimateTokens(empty.user);
    const room = CONTEXT_WINDOW - fixedTokens - role.max_tokens - TEMPLATE_MARGIN_TOKENS;
    const budget = Math.max(0, Math.min(excerptBudget, room));
    const excerpt = trimToTokenBudget(input.fit.excerpt || "", budget);
    return { ...input.fit.apply(excerpt), excerptTokens: estimateTokens(excerpt) };
  };

  let excerptBudget = EXCERPT_TOKEN_BUDGET;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    if (signal?.aborted) throw new RoleAbortedError();
    const { user, context, excerptTokens } = render(excerptBudget);
    const useJsonMode = Boolean(role.json_schema) && jsonModeAvailable;
    const request = {
      messages: [
        { role: "system", content: composeSystemPrompt(roleName, context) },
        { role: "user", content: user },
      ],
      temperature: role.temperature,
      top_p: role.top_p,
      repetition_penalty: role.repetition_penalty,
      max_tokens: role.max_tokens,
      extra_body: { enable_thinking: false },
      stream: false,
    };
    if (useJsonMode) {
      request.response_format = { type: "json_object", schema: JSON.stringify(role.json_schema) };
    }

    let completion;
    try {
      completion = await engine.chat.completions.create(request);
    } catch (error) {
      if (signal?.aborted) throw new RoleAbortedError();
      if (isContextOverflow(error) && input.fit) {
        excerptBudget = Math.floor(Math.min(excerptBudget, excerptTokens) / 2);
        continue;
      }
      if (useJsonMode && isGrammarError(error)) {
        jsonModeAvailable = false;
        console.warn("kobajinAI: JSON response_format is unavailable; falling back to strict JSON.parse", error);
        continue;
      }
      throw error;
    }
    if (signal?.aborted) throw new RoleAbortedError();

    const choice = completion.choices?.[0];
    const text = cleanModelText(choice?.message?.content || "");
    const result = {
      role: roleName,
      text,
      finishReason: choice?.finish_reason || null,
      jsonMode: useJsonMode,
      excerptTokens,
    };
    if (role.json_schema) result.json = parseJsonOutput(text);
    return result;
  }
  throw new Error(`Role ${roleName} could not fit inside the ${CONTEXT_WINDOW}-token context window`);
}

function validSeed(json) {
  if (!json || typeof json !== "object" || Array.isArray(json)) return null;
  if (typeof json.angle !== "string" || !json.angle.trim()) return null;
  if (!Array.isArray(json.facts)) return null;
  const facts = json.facts.filter((fact) => typeof fact === "string" && fact.trim()).map((fact) => fact.trim());
  if (facts.length === 0) return null;
  return { angle: json.angle.trim().slice(0, 40), facts: facts.slice(0, 2) };
}

function validVerdict(json) {
  const verdict = json && typeof json === "object" ? json.verdict : null;
  return ["supported", "unsupported", "unknown"].includes(verdict) ? verdict : null;
}

function seedUser(article, excerpt) {
  return `記事: ${article.title}\n抜粋:\n${excerpt}`;
}

function tweetUser(article, seed) {
  return [
    `題材: ${article.title}`,
    `angle: ${seed.angle}`,
    "facts:",
    ...seed.facts.map((fact) => `- ${fact}`),
  ].join("\n");
}

function factCheckUser(excerpt, text) {
  return `抜粋:\n${excerpt}\n\n投稿:\n${text}`;
}

/**
 * Seed → tweet → fact (rule, then model) / identity / novelty gates.
 * Failed drafts are regenerated up to twice; after that the article is dropped.
 * Returns { accepted, text?, log } where log holds no hidden reasoning.
 */
export async function composeTweet(engine, article, { recentTexts = [], modelId, signal } = {}) {
  const log = {
    url: article.url,
    title: article.title,
    seed: null,
    verdicts: [],
    retries: 0,
    model_id: modelId,
    json_mode: null,
  };
  let seed = null;
  let seedJsonMode = null;

  for (let attempt = 0; attempt < MAX_TWEET_ATTEMPTS; attempt += 1) {
    log.retries = attempt;
    const verdict = { attempt };
    log.verdicts.push(verdict);

    if (!seed) {
      const seedResult = await runRole(engine, "seed", {
        fit: { excerpt: article.excerpt, apply: (excerpt) => ({ user: seedUser(article, excerpt), context: "" }) },
      }, { signal });
      seedJsonMode = seedResult.jsonMode;
      const parsed = validSeed(seedResult.json);
      const facts = parsed ? filterSupportedFacts(parsed.facts, article.excerpt, article.title) : [];
      verdict.seed = parsed ? (facts.length ? "ok" : "facts_unsupported") : "invalid_json";
      if (!parsed || facts.length === 0) continue;
      seed = { angle: parsed.angle, facts };
      log.seed = seed;
    }

    const memoryContext = getMemoryContext(`${article.title} ${seed.facts.join(" ")}`);
    const tweet = await runRole(engine, "tweet", { user: tweetUser(article, seed), context: memoryContext }, { signal });
    const text = tweet.text;
    verdict.draft = text;

    const form = formGate(text, tweet.finishReason);
    verdict.form = form.pass ? "pass" : form.reason;
    if (!form.pass) continue;

    const factRule = factRuleGate(text, article.excerpt, article.title);
    verdict.fact_rule = factRule.pass ? "pass" : { missing: factRule.missing };
    if (!factRule.pass) continue;

    const identity = identityGate(text, findMemoryRecords(text));
    verdict.identity = identity.pass ? (identity.backed ? "backed_by_memory" : "pass") : { hits: identity.hits };
    if (!identity.pass) continue;

    const novelty = noveltyGate(text, recentTexts, article.excerpt);
    verdict.novelty = novelty.pass
      ? "pass"
      : { max_jaccard: Number(novelty.maxJaccard.toFixed(3)), copy_chars: novelty.copyLength };
    if (!novelty.pass) continue;

    const check = await runRole(engine, "fact_check", {
      fit: { excerpt: article.excerpt, apply: (excerpt) => ({ user: factCheckUser(excerpt, text), context: "" }) },
    }, { signal });
    const factModel = validVerdict(check.json);
    log.json_mode = { seed: seedJsonMode, fact_check: check.jsonMode };
    verdict.fact_model = factModel || "invalid_json";
    if (factModel !== "supported") continue;

    return { accepted: true, text, log };
  }
  if (log.json_mode === null) log.json_mode = { seed: seedJsonMode, fact_check: null };
  return { accepted: false, log };
}

function formatThreadLine(speaker, content, timestamp) {
  const date = new Date(timestamp || Date.now());
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  return `${month}-${day} ${hour}:${minute} 【${speaker}】${content}`;
}

const THREAD_TOKEN_BUDGET = 900;

function threadLog(post, replies) {
  const lines = replies.map(({ role, content, timestamp }) => (
    formatThreadLine(role === "assistant" ? "自分" : "参加者A", content, timestamp)
  ));
  // Keep the newest turns; the original post always leads the log.
  const head = formatThreadLine("自分", post.text, post.createdAt);
  const kept = [];
  let used = estimateTokens(head);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const cost = estimateTokens(lines[index]);
    if (kept.length > 0 && used + cost > THREAD_TOKEN_BUDGET) break;
    kept.unshift(lines[index]);
    used += cost;
  }
  return [head, ...kept].join("\n");
}

/**
 * mode=reply: answer the latest reply in a feed thread. The original post and
 * its source excerpt ride in the system context; the excerpt is trimmed first.
 */
export async function composeReply(engine, post, replies, { signal } = {}) {
  const latest = [...replies].reverse().find((reply) => reply.role === "user");
  const memoryContext = getMemoryContext(latest?.content || "");
  const user = threadLog(post, replies);
  const apply = (excerpt) => ({
    user,
    context: [
      memoryContext,
      `元の呟き: ${post.text}`,
      `元呟きの出典抜粋（Wikipedia「${post.source.title}」）:\n${excerpt}`,
    ].filter(Boolean).join("\n\n"),
  });

  const checks = [];
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const result = await runRole(engine, "reply", { fit: { excerpt: post.source.excerpt, apply } }, { signal });
    const text = result.text;
    const identity = identityGate(text, findMemoryRecords(text));
    const overlap = topicOverlap(text, `${post.text} ${post.source.title} ${latest?.content || ""}`);
    checks.push({ attempt, identity: identity.pass ? "pass" : { hits: identity.hits }, topic_overlap: Number(overlap.toFixed(3)) });
    if (text && identity.pass) return { text, checks };
  }
  return { text: CHAT_FALLBACK, checks, fallback: true };
}

/**
 * Serializes access to the single engine. Feed jobs marked interruptible are
 * stopped (engine.interruptGenerate) when chat asks for the engine.
 */
export function createEngineScheduler() {
  let tail = Promise.resolve();
  let active = null;
  let priorityWaiting = 0;
  let interrupt = () => {};

  return {
    setInterrupt(fn) { interrupt = fn; },
    get busy() { return active !== null; },
    get priorityWaiting() { return priorityWaiting > 0; },
    async acquire(kind, { interruptible = false, priority = false } = {}) {
      if (priority) {
        priorityWaiting += 1;
        if (active?.interruptible) {
          active.controller.abort();
          try { await interrupt(); } catch { /* the job still sees its aborted signal */ }
        }
      }
      const controller = new AbortController();
      let releaseTicket;
      const ticket = new Promise((resolve) => { releaseTicket = resolve; });
      const previous = tail;
      tail = previous.then(() => ticket);
      await previous;
      if (priority) priorityWaiting -= 1;
      const job = { kind, controller, interruptible };
      active = job;
      let released = false;
      return {
        signal: controller.signal,
        release() {
          if (released) return;
          released = true;
          if (active === job) active = null;
          releaseTicket();
        },
      };
    },
  };
}
