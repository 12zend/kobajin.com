import { llm } from "./llm.js";
import { POST_KINDS, findReaction } from "./prompts.js";
import { DETAIL } from "./reply_decision.js";
import { HARSH, rejectReason } from "./safety.js";
import { chars, pick, sleep } from "./util.js";

// Brain: one generation at a time, visitor work first.
// A single WebLLM engine handles one request at a time, so every call
// goes through this queue. Visitor reactions jump ahead of background
// posts, kobajin-to-kobajin threads, and trends.
//
// So the real model does not keep the GPU busy back to back, background
// jobs wait BACKGROUND_REST_MS after each kobajinAI generation, and wait
// while the tab is hidden. Visitor jobs never wait.

export const PRIORITY = { visitor: 2, background: 1 };
const BACKGROUND_REST_MS = 2500;
const queue = [];
let queueRunning = false;
let queueSeq = 0;
let restUntil = 0;

const canRun = () => queue[0]?.priority === PRIORITY.visitor || (!document.hidden && Date.now() >= restUntil);

function enqueue(priority, run) {
  return new Promise((resolve, reject) => {
    queue.push({ priority, seq: queueSeq++, run, resolve, reject });
    queue.sort((a, b) => b.priority - a.priority || a.seq - b.seq);
    pumpQueue();
  });
}

async function pumpQueue() {
  if (queueRunning) return;
  if (!queue.length) { document.body.dataset.busy = "false"; return; }
  queueRunning = true;
  if (!canRun()) {
    document.body.dataset.busy = "false";
    while (!canRun()) await sleep(250);
  }
  const job = queue.shift();
  document.body.dataset.busy = "true";
  const real = llm.name !== "mock";
  try { job.resolve(await job.run()); } catch (error) { job.reject(error); }
  if (real) restUntil = Date.now() + BACKGROUND_REST_MS;
  queueRunning = false;
  pumpQueue();
}

// The first line, without think blocks, speaker labels, log stamps, or
// wrapping quotes.
function firstLine(raw) {
  const text = String(raw || "")
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<\/?think>/g, "")
    .split("\n").map((line) => line.trim()).find(Boolean) || "";
  return text
    .replace(/^\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}\s*/, "")
    .replace(/^\s*(【[^】]*】|[^\s:：]{1,12}[:：])\s*/, "")
    .replace(/^[「『"']+|[」』"']+$/g, "")
    .trim();
}

// Fits a line into maxChars. A long line is cut where a sentence or
// clause ends, never mid-word; with no such place it does not fit
// (null), unless `force` (the last try) allows a plain cut.
function fit(text, maxChars, force = false) {
  const list = chars(text);
  if (list.length <= maxChars) return text;
  const head = list.slice(0, maxChars + 1).join("");
  const cut = Math.max(...["。", "！", "？", "!", "?", "、", " ", "　", "…"].map((mark) => head.lastIndexOf(mark)));
  if (cut >= Math.floor(maxChars / 2)) return chars(head.slice(0, cut + 1)).slice(0, maxChars).join("").replace(/[、\s　]+$/, "");
  return force ? list.slice(0, maxChars).join("") : null;
}

// Generates until a line passes the gates (safety.js), up to `tries`
// times; "" when none does. `source` names what wrote the shown line.
// `input` and `recent` are functions, read when each try runs.
async function generate(priority, task, input, { maxChars, recent = () => [], echoOf = "", tries = 3 }) {
  for (let attempt = 0; attempt < tries; attempt += 1) {
    const raw = await enqueue(priority, () => llm.complete(task, input()));
    const source = llm.source;
    const line = fit(firstLine(raw), maxChars, attempt === tries - 1);
    if (line === null) continue;
    const reason = rejectReason(line, { recent: recent(), source: echoOf });
    if (!reason) return { text: line, source };
    if (import.meta.env.DEV) console.debug(`[brain] ${task} rejected (${reason}):`, line);
  }
  return { text: "", source: llm.source };
}


const recentKinds = [];

export const brain = {
  // An account posts a kind it leans toward (accounts.js), most of the
  // time, among the kinds the timeline has not used lately.
  // `timeline` is what everyone posted lately, so posts are not repeats.
  async post(account, recent, timeline = []) {
    const open = POST_KINDS.filter((k) => !recentKinds.includes(k));
    const leaning = (account.kinds || []).filter((k) => open.includes(k));
    const kind = pick(leaning.length && Math.random() < 0.7 ? leaning : open);
    recentKinds.push(kind);
    if (recentKinds.length > 4) recentKinds.shift();
    const input = { recent, kind, persona: account.persona, voice: account.voice };
    const result = await generate(PRIORITY.background, "post", () => input, { maxChars: 24, recent: () => [...recent, ...timeline] });
    return { ...result, text: result.text.replace(/#\S+/g, "").trim() };
  },
  // `others` is read when the job runs, so replies queued together still
  // see the ones that finished before them.
  // A request to say more (具体的に, 詳しく, たとえば？, どういうこと) about
  // the account's own line gets a longer, concrete answer instead of a
  // 10-character reaction.
  async reply(account, { text, parent = null, addressed = false }, others, priority = PRIORITY.visitor) {
    const detail = Boolean(addressed && parent && DETAIL.test(text));
    const task = detail ? "elaborate" : "reply";
    const input = { text, parent, addressed, persona: account.persona, voice: account.voice };
    return generate(priority, task, () => ({ ...input, others: others() }), { maxChars: detail ? 25 : 12, recent: others, echoOf: text });
  },
  // One emoji for the post; anything off the list falls back to 🤔.
  async react(text) {
    const raw = await enqueue(PRIORITY.visitor, () => llm.complete("react", { text }));
    return findReaction(raw) || "🤔";
  },
  // The kobajinAI versions answer 「いい」 more often than 「はい」, even to
  // insults, so an insult is never liked whatever the model says.
  async judge(text) {
    if (HARSH.test(text)) return false;
    const raw = await enqueue(PRIORITY.visitor, () => llm.complete("judge", { text }));
    return /はい|いい|yes|good|👍/i.test(String(raw)) && !/いいえ|よくない|良くない/.test(String(raw));
  },
  async trends(count) {
    const raw = await enqueue(PRIORITY.background, () => llm.complete("trend", { count }));
    const tags = String(raw).split(/\n|\s+(?=#)/)
      .map((line) => line.trim().replace(/^[-・\d.、)\s]+/, ""))
      .filter(Boolean)
      .map((line) => `#${line.replace(/^[#＃]+/, "").replace(/\s+/g, "")}`)
      .filter((tag) => chars(tag).length > 1 && chars(tag).length <= 13)
      .filter((tag) => !rejectReason(tag.slice(1)));
    return [...new Set(tags)].slice(0, count);
  },
};
