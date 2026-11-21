import { ACCOUNTS } from "./accounts.js";
import { REACTION_EMOJI } from "./prompts.js";
import { adoptPost, posts } from "./state.js";

// Saves the timeline in this browser, so a visit picks up where the last
// one left off. Typing and unsent drafts are not saved. Older posts drop
// off past MAX_POSTS, except bookmarked ones and the visitor's own.

const KEY = "kobajinverse.timeline.v1";
const MAX_POSTS = 80;
let timer = 0;

const known = (id) => Boolean(ACCOUNTS[id]);
// Reactions saved before the six emoji (prompts.js) are dropped.
const allowed = (emoji) => REACTION_EMOJI.includes(emoji);

function serialize(post) {
  return {
    id: post.id,
    authorId: post.authorId,
    text: post.text,
    createdAt: post.createdAt,
    source: post.source || "",
    reactions: [...post.reactions].map(([emoji, who]) => [emoji, [...who]]),
    reposts: [...post.reposts],
    bookmarked: post.bookmarked,
    replies: post.replies.map(({ authorId, toId, text, createdAt, source, reactions }) => ({ authorId, toId, text, createdAt, source: source || "", ...(reactions ? { reactions } : {}) })),
  };
}

// Reactions on a reply: { emoji: [accountId, ...] }, known accounts only.
function reviveReply(reply) {
  const reactions = Object.fromEntries(Object.entries(reply.reactions && typeof reply.reactions === "object" ? reply.reactions : {})
    .map(([emoji, who]) => [emoji, allowed(emoji) && Array.isArray(who) ? who.filter(known) : []])
    .filter(([, who]) => who.length));
  const { reactions: _, ...rest } = reply;
  return Object.keys(reactions).length ? { ...rest, reactions } : rest;
}

// Lines the mock wrote on earlier visits are dropped: the site shows only
// what kobajinAI wrote (llm.js).
const fromModel = (line) => line.source !== "mock";

function revive(data) {
  if (!data || !/^p\d+$/.test(data.id) || !known(data.authorId) || typeof data.text !== "string" || !fromModel(data)) return null;
  return {
    id: data.id,
    authorId: data.authorId,
    text: data.text,
    createdAt: Number(data.createdAt) || Date.now(),
    source: data.source || "",
    reactions: new Map((data.reactions || []).filter(([emoji]) => allowed(emoji)).map(([emoji, who]) => [emoji, new Set((who || []).filter(known))]).filter(([, who]) => who.size)),
    reposts: new Set((data.reposts || []).filter(known)),
    bookmarked: Boolean(data.bookmarked),
    replies: (data.replies || []).filter((reply) => known(reply.authorId) && known(reply.toId) && typeof reply.text === "string" && fromModel(reply)).map(reviveReply),
    drafts: [],
    typing: new Set(),
  };
}

export function saveNow() {
  clearTimeout(timer);
  const sorted = [...posts.values()].sort((a, b) => b.createdAt - a.createdAt);
  const kept = sorted.filter((post, index) => index < MAX_POSTS || post.bookmarked || post.authorId === "visitor");
  try {
    localStorage.setItem(KEY, JSON.stringify({ savedAt: Date.now(), posts: kept.map(serialize) }));
  } catch { /* storage full or blocked: this visit only */ }
}

export function scheduleSave() {
  clearTimeout(timer);
  timer = setTimeout(saveNow, 500);
}

// Loads the saved timeline into state. Returns its posts (newest first)
// and when it was last saved, i.e. roughly when the visitor left.
export function restoreTimeline() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(KEY) || "null"); } catch { /* start fresh */ }
  if (!saved || !Array.isArray(saved.posts)) return { posts: [], savedAt: 0 };
  const list = saved.posts.map(revive).filter(Boolean).sort((a, b) => b.createdAt - a.createdAt);
  list.forEach(adoptPost);
  return { posts: list, savedAt: Number(saved.savedAt) || 0 };
}

export function clearTimeline() {
  try { localStorage.removeItem(KEY); } catch { /* nothing saved */ }
}

// Leaving the page records when the visit ended.
addEventListener("pagehide", saveNow);
document.addEventListener("visibilitychange", () => { if (document.hidden) saveNow(); });
