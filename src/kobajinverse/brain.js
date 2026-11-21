import { llm } from "./llm.js";
import { POST_KINDS, REACTION_EMOJI } from "./prompts.js";
import { chars, pick } from "./util.js";

// Brain: one generation at a time, visitor work first.
// A single WebLLM engine handles one request at a time, so every call
// goes through this queue. Visitor reactions jump ahead of background
// posts and trends.

const PRIORITY = { visitor: 2, background: 1 };
const queue = [];
let queueRunning = false;
let queueSeq = 0;

function enqueue(priority, run) {
  return new Promise((resolve, reject) => {
    queue.push({ priority, seq: queueSeq++, run, resolve, reject });
    queue.sort((a, b) => b.priority - a.priority || a.seq - b.seq);
    pumpQueue();
  });
}

async function pumpQueue() {
  if (queueRunning) return;
  const job = queue.shift();
  if (!job) { document.body.dataset.busy = "false"; return; }
  queueRunning = true;
  document.body.dataset.busy = "true";
  try { job.resolve(await job.run()); } catch (error) { job.reject(error); }
  queueRunning = false;
  pumpQueue();
}

function clean(raw, maxChars) {
  const text = String(raw || "")
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/^\s*(【[^】]*】|[^\s:：]{1,12}[:：])\s*/, "")
    .split("\n").map((line) => line.trim()).find(Boolean) || "";
  const unquoted = text.replace(/^[「『"']+|[」』"']+$/g, "").trim();
  const list = chars(unquoted);
  return list.length > maxChars ? list.slice(0, maxChars).join("") : unquoted;
}

const DETAIL = /具体|詳し|くわし|kwsk|詳細|例えば|たとえば|例は|例を|どういう(こと|意味)|どゆこと|どういうことや|つまり|説明して|掘り下げ|もっと(教えて|聞かせて|言って)|もう(少し|ちょい|ちょっと)(詳|くわ|教|説明)/i;

const recentKinds = [];

export const brain = {
  async post(recent) {
    const kind = pick(POST_KINDS.filter((k) => !recentKinds.includes(k)));
    recentKinds.push(kind);
    if (recentKinds.length > 4) recentKinds.shift();
    const raw = await enqueue(PRIORITY.background, () => llm.complete("post", { recent, kind }));
    return clean(raw, 24).replace(/#\S+/g, "").trim();
  },
  // `others` is read when the job runs, so replies queued together still
  // see the ones that finished before them.
  // A request to say more (具体的に, 詳しく, たとえば？, どういうこと) about
  // the account's own line gets a longer, concrete answer instead of a
  // 10-character reaction.
  async reply({ text, parent = null, addressed = false }, others) {
    const detail = Boolean(addressed && parent && DETAIL.test(text));
    const task = detail ? "elaborate" : "reply";
    const raw = await enqueue(PRIORITY.visitor, () => llm.complete(task, { text, parent, addressed, others: others() }));
    return clean(raw, detail ? 25 : 10);
  },
  // One emoji for the post; anything off the list falls back to 🤔.
  async react(text) {
    const raw = await enqueue(PRIORITY.visitor, () => llm.complete("react", { text }));
    return REACTION_EMOJI.find((emoji) => String(raw).includes(emoji)) || "🤔";
  },
  async judge(text) {
    const raw = await enqueue(PRIORITY.visitor, () => llm.complete("judge", { text }));
    return /はい|yes|good|👍/i.test(String(raw)) && !/いいえ/.test(String(raw));
  },
  async trends(count) {
    const raw = await enqueue(PRIORITY.background, () => llm.complete("trend", { count }));
    const tags = String(raw).split(/\n|\s+(?=#)/)
      .map((line) => line.trim().replace(/^[-・\d.、)\s]+/, ""))
      .filter(Boolean)
      .map((line) => `#${line.replace(/^[#＃]+/, "").replace(/\s+/g, "")}`)
      .filter((tag) => chars(tag).length > 1 && chars(tag).length <= 13);
    return [...new Set(tags)].slice(0, count);
  },
};
