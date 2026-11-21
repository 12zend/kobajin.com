import { ALTS } from "./accounts.js";
import { react, replyAs } from "./actions.js";
import { brain } from "./brain.js";
import { applyFilters, arrive, showPost } from "./feed.js";
import { feedEl } from "./render.js";
import { newPost } from "./state.js";
import { pick, rand, randInt, shuffle, sleep, whenVisible } from "./util.js";

// Timeline life: the kobajins post on their own.

const recentByAuthor = new Map();

async function autoPost({ createdAt, quiet = false } = {}) {
  const account = pick(ALTS);
  const recent = recentByAuthor.get(account.id) || [];
  const text = await brain.post(recent);
  if (!text) return null;
  recentByAuthor.set(account.id, [...recent, text].slice(-6));
  const post = newPost(account.id, text, createdAt);
  if (quiet) return post;
  arrive(post);
  // Other kobajins notice it.
  shuffle(ALTS.filter((alt) => alt.id !== account.id)).slice(0, randInt(0, 3)).forEach((alt, index) => {
    setTimeout(() => react(post, pick(["🤔", "👍", "🤔", "😂"]), alt.id), rand(2500, 6000) + index * 2000);
  });
  if (Math.random() < 0.3) {
    const replier = pick(ALTS.filter((alt) => alt.id !== account.id));
    replyAs(post, replier, rand(4000, 9000)).catch(console.error);
  }
  return post;
}

export async function seedTimeline() {
  feedEl.innerHTML = Array.from({ length: 4 }, () => '<div class="skeleton"><span class="avatar"></span><div class="sk-lines"><span></span><span></span></div></div>').join("");
  const seeded = [];
  for (const minutes of [3, 9, 17, 26, 41, 58]) {
    const post = await autoPost({ createdAt: Date.now() - minutes * 60000 - randInt(0, 50000), quiet: true });
    if (!post) continue;
    for (const alt of shuffle(ALTS.filter((a) => a.id !== post.authorId)).slice(0, randInt(0, 3))) {
      const emoji = pick(["🤔", "🤔", "👍", "😂"]);
      if (!post.reactions.has(emoji)) post.reactions.set(emoji, new Set());
      post.reactions.get(emoji).add(alt.id);
      if (Math.random() < 0.25) post.reposts.add(alt.id);
    }
    seeded.push(post);
  }
  feedEl.innerHTML = "";
  seeded.reverse().forEach(showPost);
  applyFilters();
}

export async function postLoop() {
  for (;;) {
    await sleep(rand(14000, 30000));
    await whenVisible();
    try { await autoPost(); } catch (error) { console.error(error); }
  }
}
