import { ALTS } from "./accounts.js";
import { banter, react } from "./actions.js";
import { brain } from "./brain.js";
import { llm } from "./llm.js";
import { applyFilters, arrive, showPost } from "./feed.js";
import { feedEl } from "./render.js";
import { newPost, posts } from "./state.js";
import { pick, rand, randInt, shuffle, sleep, whenVisible } from "./util.js";

// Timeline life: the kobajins post on their own and talk to each other.

const recentByAuthor = new Map();
const timelineTexts = () => [...posts.values()].sort((a, b) => b.createdAt - a.createdAt).slice(0, 30).map((post) => post.text);

// Seeds each account's recent lines from a restored timeline, so it does
// not post what it said last visit.
export function rememberPosts(list) {
  for (const post of [...list].sort((a, b) => a.createdAt - b.createdAt)) {
    recentByAuthor.set(post.authorId, [...(recentByAuthor.get(post.authorId) || []), post.text].slice(-6));
  }
}

// A post that already exists before the visitor sees it: a few
// reactions and reposts from others, but no live typing.
function settle(post) {
  for (const alt of shuffle(ALTS.filter((a) => a.id !== post.authorId)).slice(0, randInt(0, 3))) {
    const emoji = pick(["🤔", "🤔", "👍", "😂"]);
    if (!post.reactions.has(emoji)) post.reactions.set(emoji, new Set());
    post.reactions.get(emoji).add(alt.id);
    if (Math.random() < 0.25) post.reposts.add(alt.id);
  }
}

async function autoPost({ createdAt, quiet = false } = {}) {
  const account = pick(ALTS);
  const recent = recentByAuthor.get(account.id) || [];
  const { text, source } = await brain.post(account, recent, timelineTexts());
  if (!text) return null;
  recentByAuthor.set(account.id, [...recent, text].slice(-6));
  const post = newPost(account.id, text, createdAt, source);
  if (quiet) { settle(post); return post; }
  arrive(post);
  // Other kobajins notice it.
  shuffle(ALTS.filter((alt) => alt.id !== account.id)).slice(0, randInt(0, 3)).forEach((alt, index) => {
    setTimeout(() => react(post, pick(["🤔", "👍", "🤔", "😂"]), alt.id), rand(2500, 6000) + index * 2000);
  });
  if (Math.random() < (llm.name === "mock" ? 0.45 : 0.3)) banter(post).catch(console.error);
  return post;
}

const skeletons = (n) => Array.from({ length: n }, () => '<div class="skeleton"><span class="avatar"></span><div class="sk-lines"><span></span><span></span></div></div>').join("");

export async function seedTimeline() {
  feedEl.innerHTML = skeletons(4);
  const seeded = [];
  for (const minutes of [3, 9, 17, 26, 41, 58]) {
    const post = await autoPost({ createdAt: Date.now() - minutes * 60000 - randInt(0, 50000), quiet: true });
    if (post) seeded.push(post);
  }
  feedEl.innerHTML = "";
  seeded.reverse().forEach(showPost);
  applyFilters();
}

// Back after a while: the kobajins posted in the meantime. The pages
// cannot run while closed, so these are written now, dated between the
// last visit and now, and shown above a divider.
const AWAY_MIN_MS = 3 * 60000;
export async function catchUp(since) {
  const away = Date.now() - since;
  if (!since || away < AWAY_MIN_MS) return 0;
  const count = Math.min(4, Math.max(1, Math.floor(away / (10 * 60000))));
  const loading = document.createElement("div");
  loading.innerHTML = skeletons(Math.min(count, 2));
  feedEl.prepend(loading);
  const times = Array.from({ length: count }, () => since + rand(0.1, 0.95) * away).sort((a, b) => a - b);
  const made = [];
  for (const createdAt of times) {
    try {
      const post = await autoPost({ createdAt, quiet: true });
      if (post) made.push(post);
    } catch (error) { console.error(error); }
  }
  loading.remove();
  if (!made.length) return 0;
  const divider = document.createElement("div");
  divider.className = "away-divider";
  divider.innerHTML = "<span>ここから前回の訪問までのポスト</span>";
  feedEl.prepend(divider);
  made.forEach(showPost);
  return made.length;
}

// The mock posts every 14–30 s. kobajinAI posts about half as often:
// each post brings replies and threads, all generated on the GPU.
export async function postLoop() {
  for (;;) {
    await sleep(llm.name === "mock" ? rand(14000, 30000) : rand(35000, 65000));
    await whenVisible();
    try { await autoPost(); } catch (error) { console.error(error); }
  }
}
