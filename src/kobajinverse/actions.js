import { ACCOUNTS, ALTS, SELF, VISITOR } from "./accounts.js";
import { PRIORITY, brain } from "./brain.js";
import { say } from "./dom.js";
import { notify } from "./notifications.js";
import { refresh, renderUsers } from "./render.js";
import { typingCounts } from "./state.js";
import { rand, randInt, shuffle, sleep } from "./util.js";

// Actions: reactions, reposts, typing, and replies, and how the kobajins
// answer what the visitor writes and talk among themselves.

// Anything a kobajin does to the visitor's post or reply is a
// notification (notifications.js).
const toVisitor = (post, accountId) => post.authorId === VISITOR.id && accountId !== VISITOR.id;

export function react(post, emoji, accountId, { toggle = false } = {}) {
  if (!post.reactions.has(emoji)) post.reactions.set(emoji, new Set());
  const who = post.reactions.get(emoji);
  if (toggle && who.has(accountId)) who.delete(accountId);
  else if (who.has(accountId)) return;
  else {
    who.add(accountId);
    if (toVisitor(post, accountId)) notify({ type: "reaction", post, actorId: accountId, emoji });
  }
  if (!who.size) post.reactions.delete(emoji);
  refresh(post, { bumped: new Set([emoji]) });
}

export function repost(post, accountId, { toggle = false } = {}) {
  if (toggle && post.reposts.has(accountId)) post.reposts.delete(accountId);
  else {
    if (!post.reposts.has(accountId) && toVisitor(post, accountId)) notify({ type: "repost", post, actorId: accountId });
    post.reposts.add(accountId);
  }
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
function addReply(post, accountId, text, toId = post.authorId, source = "") {
  const reply = { authorId: accountId, toId, text, createdAt: Date.now(), source };
  post.replies.push(reply);
  if (toId === VISITOR.id && accountId !== VISITOR.id) notify({ type: "reply", post, actorId: accountId, text });
  refresh(post, { freshReply: post.replies.length - 1 });
  return reply;
}

// Mentions are addressing, not content: the model should not echo a handle.
const MENTION = /@([A-Za-z0-9_]+)/g;
const withoutMentions = (text) => text.replace(MENTION, "").replace(/\s+/g, " ").trim() || text;
const mentionedBots = (text) => [...new Set([...text.matchAll(MENTION)].map((m) => m[1]))]
  .map((handle) => [SELF, ...ALTS].find((account) => account.handle === handle)).filter(Boolean);

// `reply` is the reply being answered ({ authorId, toId, text,
// parentText }); without it the account answers the post itself.
// Resolves to the reply it added, or null.
export async function replyAs(post, account, delay, reply = null, priority = PRIORITY.visitor) {
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
    const { text, source } = await brain.reply(account, prompt, () => [...post.replies.map((r) => r.text), ...post.drafts], priority);
    if (text) post.drafts.push(text);
    await shown;
    if (!text) return null;
    post.drafts.splice(post.drafts.indexOf(text), 1);
    return addReply(post, account.id, text, reply ? reply.authorId : post.authorId, source);
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
  })().catch(console.error);

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
  })().catch(console.error);

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

// kobajins talk among themselves under a post: someone replies, the
// author often answers back (an addressed reply, so a question gets an
// answer), and the exchange sometimes goes a round or two more, or a
// third kobajin cuts in. Background priority, so the visitor's own
// replies never wait behind it.
export async function banter(post, { first = rand(4000, 9000) } = {}) {
  const author = ACCOUNTS[post.authorId];
  const replier = shuffle(ALTS.filter((alt) => alt.id !== author.id))[0];
  const opening = await replyAs(post, replier, first, null, PRIORITY.background);
  if (!opening) return;
  let last = { reply: opening, parentText: post.text };
  let speakers = [author, replier];
  // Each round: the one who was answered talks back, with fading odds.
  for (const chance of [0.7, 0.45, 0.25]) {
    if (Math.random() > chance) break;
    const [next] = speakers;
    const answer = await replyAs(post, next, rand(2500, 6000), { ...last.reply, parentText: last.parentText }, PRIORITY.background);
    if (!answer) break;
    last = { reply: answer, parentText: last.reply.text };
    speakers = [speakers[1], speakers[0]];
  }
  // Now and then someone else chimes in on the last line.
  if (Math.random() < 0.25) {
    const outsider = shuffle(ALTS.filter((alt) => alt.id !== author.id && alt.id !== replier.id))[0];
    await replyAs(post, outsider, rand(3000, 7000), { ...last.reply, parentText: last.parentText }, PRIORITY.background);
  }
}
