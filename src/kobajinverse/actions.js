import { ACCOUNTS, ALTS, SELF, VISITOR } from "./accounts.js";
import { brain } from "./brain.js";
import { say } from "./dom.js";
import { refresh, renderUsers } from "./render.js";
import { typingCounts } from "./state.js";
import { rand, randInt, shuffle, sleep } from "./util.js";

// Actions: reactions, reposts, typing, and replies, and how the kobajins
// answer what the visitor writes.

export function react(post, emoji, accountId, { toggle = false } = {}) {
  if (!post.reactions.has(emoji)) post.reactions.set(emoji, new Set());
  const who = post.reactions.get(emoji);
  if (toggle && who.has(accountId)) who.delete(accountId);
  else if (who.has(accountId)) return;
  else who.add(accountId);
  if (!who.size) post.reactions.delete(emoji);
  refresh(post, { bumped: new Set([emoji]) });
}

export function repost(post, accountId, { toggle = false } = {}) {
  if (toggle && post.reposts.has(accountId)) post.reposts.delete(accountId);
  else post.reposts.add(accountId);
  refresh(post);
}

function setTyping(post, accountId, on) {
  if (on) post.typing.add(accountId); else post.typing.delete(accountId);
  typingCounts.set(accountId, Math.max(0, (typingCounts.get(accountId) || 0) + (on ? 1 : -1)));
  refresh(post);
  renderUsers();
}

// Each reply records who it answers (toId), so the thread shows the
// right 返信先 and replies to replies keep their context.
function addReply(post, accountId, text, toId = post.authorId) {
  post.replies.push({ authorId: accountId, toId, text, createdAt: Date.now() });
  refresh(post, { freshReply: post.replies.length - 1 });
}

// Mentions are addressing, not content: the model should not echo a handle.
const MENTION = /@([A-Za-z0-9_]+)/g;
const withoutMentions = (text) => text.replace(MENTION, "").replace(/\s+/g, " ").trim() || text;
const mentionedBots = (text) => [...new Set([...text.matchAll(MENTION)].map((m) => m[1]))]
  .map((handle) => [SELF, ...ALTS].find((account) => account.handle === handle)).filter(Boolean);

// `reply` is the reply being answered; without it the account answers
// the post itself.
export async function replyAs(post, account, delay, reply = null) {
  await sleep(delay);
  setTyping(post, account.id, true);
  try {
    const shown = sleep(rand(900, 1800));
    const prompt = reply
      ? {
        text: withoutMentions(reply.text),
        parent: withoutMentions(reply.parentText),
        addressed: reply.toId === account.id || mentionedBots(reply.text).includes(account),
      }
      : { text: withoutMentions(post.text) };
    const text = await brain.reply(prompt, () => [...post.replies.map((r) => r.text), ...post.drafts]);
    if (text) post.drafts.push(text);
    await shown;
    if (text) {
      post.drafts.splice(post.drafts.indexOf(text), 1);
      addReply(post, account.id, text, reply ? reply.authorId : post.authorId);
    }
  } finally {
    setTyping(post, account.id, false);
  }
}

// The visitor posts. kobajinAI picks an emoji that fits the post;
// kobajin reacts with it first after a short, uneven pause, then the
// other kobajins pile on one by one, now and then with 👀 instead.
// kobajinAI judges the post; if it likes it, 👍 shows up and grows, and
// some fans repost it. A few short replies arrive too.
export function choreograph(post) {
  const thinkers = shuffle(ALTS).slice(0, randInt(2, ALTS.length));
  (async () => {
    const [emoji] = await Promise.all([brain.react(withoutMentions(post.text)), sleep(rand(1100, 2600))]);
    react(post, emoji, SELF.id);
    for (const account of thinkers) {
      await sleep(rand(700, 2800));
      react(post, Math.random() < 0.8 ? emoji : "👀", account.id);
    }
  })();

  (async () => {
    const liked = await brain.judge(withoutMentions(post.text));
    if (!liked) return;
    const fans = shuffle([SELF, ...ALTS]).slice(0, randInt(1, 4));
    await sleep(rand(1800, 3200));
    for (const account of fans) {
      react(post, "👍", account.id);
      if (Math.random() < 0.4) setTimeout(() => repost(post, account.id), rand(800, 2500));
      await sleep(rand(600, 2200));
    }
    say(`kobajinたちが ${fans.length} 件 👍 しました`);
  })();

  // Anyone the post @mentions answers first.
  const mentioned = mentionedBots(post.text);
  const others = shuffle(ALTS.filter((alt) => !mentioned.includes(alt))).slice(0, Math.max(mentioned.length ? 1 : 2, randInt(2, 3) - mentioned.length));
  [...mentioned, ...others].forEach((account, index) => {
    replyAs(post, account, rand(1800, 3000) + index * rand(1400, 2600)).catch(console.error);
  });
}

// The visitor replies in a thread. Whoever they answered replies back;
// answering their own post brings in one or two others instead.
export function visitorReply(post, toId, parentText, text) {
  addReply(post, VISITOR.id, text, toId);
  const reply = { authorId: VISITOR.id, toId, text, parentText };
  const target = ACCOUNTS[toId];
  const mentioned = mentionedBots(text).filter((account) => account.id !== toId);
  const responders = target.visitor
    ? [...mentioned, ...shuffle(ALTS.filter((alt) => !mentioned.includes(alt)))].slice(0, Math.max(mentioned.length, randInt(1, 2)))
    : [target, ...mentioned];
  responders.forEach((account, index) => {
    replyAs(post, account, rand(1500, 2800) + index * rand(1400, 2600), reply).catch(console.error);
  });
  if (!target.visitor && Math.random() < 0.6) {
    Promise.all([brain.react(withoutMentions(text)), sleep(rand(900, 2000))])
      .then(([emoji]) => react(post, emoji, toId))
      .catch(console.error);
  }
}
