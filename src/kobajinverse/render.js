import { ACCOUNTS, ALTS, SELF, VISITOR } from "./accounts.js";
import { $, escapeHtml } from "./dom.js";
import { following, typingCounts } from "./state.js";
import { scheduleSave } from "./store.js";
import { llm } from "./llm.js";

// Rendering: HTML for posts, threads, and the user list.

export const feedEl = $("feed");

const linkTags = (text) => escapeHtml(text).replace(/([#＃][^\s#＃]+|@[A-Za-z0-9_]+)/g, '<span class="tag">$1</span>');
export const icon = (name) => `<svg class="i" aria-hidden="true"><use href="#i-${name}" /></svg>`;

export function avatar(account, size = "", online = false) {
  const classes = `avatar ${size}${account.visitor ? " visitor" : ""}${online ? " online" : ""}`;
  if (account.visitor) {
    const style = account.color ? ` style="background:${account.color}"` : "";
    return account.emoji
      ? `<span class="${classes} has-emoji"${style} aria-hidden="true"><span class="face">${account.emoji}</span></span>`
      : `<span class="${classes}"${style} aria-hidden="true"></span>`;
  }
  const filter = account.filter ? ` style="filter:${account.filter}"` : "";
  return `<span class="${classes}" aria-hidden="true"><img src="${account.icon}" alt="" width="40" height="40" decoding="async"${filter} /></span>`;
}

function ago(ts) {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 10) return "たった今";
  if (s < 60) return `${s}秒`;
  if (s < 3600) return `${Math.floor(s / 60)}分`;
  return `${Math.floor(s / 3600)}時間`;
}

const timeTag = (ts) => `<time data-ts="${ts}" datetime="${new Date(ts).toISOString()}">${ago(ts)}</time>`;

function badge(account) {
  if (account.visitor) return '<span class="badge you">訪問者</span>';
  if (account.role) return `<span class="badge alt">${account.role}</span>`;
  return "";
}

// While a kobajinAI version is running, lines the mock wrote earlier say
// so, and lines from another version name it.
function sourceTag(source) {
  if (!source || llm.name === "mock" || source === llm.source) return "";
  return `<span class="src-tag" title="${source === "mock" ? "モック推論の文です" : `kobajinAI ${source} の文です`}">${source === "mock" ? "モック" : source}</span>`;
}

function metaHtml(account, ts, source = "") {
  return `<div class="post-meta"><span class="post-name">${escapeHtml(account.name)}</span>${badge(account)}<span class="handle">@${account.handle}</span><span class="handle">·</span>${timeTag(ts)}${sourceTag(source)}</div>`;
}

function reactionsHtml(post, bumped) {
  const pills = [...post.reactions].filter(([, who]) => who.size).map(([emoji, who]) => {
    const names = [...who].map((id) => ACCOUNTS[id].name);
    const label = `${names.join("、")} が ${emoji} でリアクション`;
    return `<button class="rx${bumped.has(emoji) ? " bump" : ""}" type="button" data-emoji="${emoji}" aria-pressed="${who.has(VISITOR.id)}" data-who="${escapeHtml(label)}" aria-label="${escapeHtml(`${emoji} ${who.size}件。${label}`)}"><span>${emoji}</span><span class="count">${who.size}</span></button>`;
  }).join("");
  return pills ? `${pills}<button class="rx-add" type="button" data-add aria-label="リアクションを追加">${icon("react")}</button>` : "";
}

function replyHtml(post, reply, index, fresh) {
  const account = ACCOUNTS[reply.authorId];
  return `<div class="reply${fresh ? " enter" : ""}">
    ${avatar(account, "sm")}
    <div class="reply-body">
      ${metaHtml(account, reply.createdAt, reply.source)}
      <div class="reply-to">返信先: <b>@${ACCOUNTS[reply.toId].handle}</b>さん</div>
      <p class="post-text">${linkTags(reply.text)}</p>
      <button class="reply-act" type="button" data-reply-to="${index}" aria-label="${escapeHtml(account.name)}に返信">${icon("reply")}</button>
    </div>
  </div>`;
}

function typingHtml(post) {
  if (!post.typing.size) return "";
  const names = [...post.typing].map((id) => `<b>${escapeHtml(ACCOUNTS[id].name)}</b>`);
  return `<div class="typing"><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span><span>${names.join("、")} が入力中…</span></div>`;
}

// refresh() redraws only .post-content. The reply box lives in
// .reply-slot, so arriving reactions never reset what the visitor is
// typing (or break IME input).
export function postHtml(post) {
  return `${avatar(ACCOUNTS[post.authorId])}
    <div class="post-main"><div class="post-content">${postContentHtml(post)}</div><div class="reply-slot"></div></div>`;
}

function postContentHtml(post, { bumped = new Set(), freshReply = -1 } = {}) {
  const account = ACCOUNTS[post.authorId];
  const reacted = [...post.reactions.values()].reduce((sum, who) => sum + who.size, 0);
  const thread = post.replies.map((reply, index) => replyHtml(post, reply, index, index === freshReply)).join("");
  const count = (n) => `<span>${n || ""}</span>`;
  return `${metaHtml(account, post.createdAt, post.source)}
      <p class="post-text">${linkTags(post.text)}</p>
      <div class="reactions">${reactionsHtml(post, bumped)}</div>
      <div class="post-actions">
        <button class="act" type="button" data-kind="reply" aria-label="返信 ${post.replies.length}件">${icon("reply")}${count(post.replies.length)}</button>
        <button class="act" type="button" data-kind="repost" aria-pressed="${post.reposts.has(VISITOR.id)}" aria-label="リポスト ${post.reposts.size}件">${icon("repost")}${count(post.reposts.size)}</button>
        <button class="act" type="button" data-kind="react" data-add aria-label="リアクション ${reacted}件">${icon("react")}${count(reacted)}</button>
        <button class="act" type="button" data-kind="bookmark" aria-pressed="${post.bookmarked}" aria-label="ブックマーク">${icon("bookmark")}</button>
        <button class="act" type="button" data-kind="share" aria-label="共有" aria-haspopup="menu">${icon("share")}</button>
      </div>
      <div class="thread">${thread}${typingHtml(post)}</div>`;
}

export function refresh(post, options) {
  const el = feedEl.querySelector(`[data-id="${post.id}"]`);
  const content = el?.querySelector(".post-content");
  if (content) content.innerHTML = postContentHtml(post, options);
  scheduleSave();
}

// Redraws every post, e.g. after the visitor renames themself or the
// model changes (source tags). Reply boxes stay; avatars are repainted.
export function refreshAll(posts) {
  feedEl.querySelectorAll(".post").forEach((el) => {
    const post = posts.get(el.dataset.id);
    if (!post) return;
    el.querySelector(".post-content").innerHTML = postContentHtml(post);
    const face = el.querySelector(":scope > .avatar");
    if (face) face.outerHTML = avatar(ACCOUNTS[post.authorId]);
  });
  document.querySelectorAll(".reply-box .avatar, .composer > .avatar, .me-chip .avatar").forEach((el) => {
    el.outerHTML = avatar(VISITOR, el.classList.contains("sm") ? "sm" : "");
  });
}

// Keeps relative times (たった今, 3分) current.
export function startClock() {
  setInterval(() => {
    document.querySelectorAll("time[data-ts]").forEach((el) => { el.textContent = ago(Number(el.dataset.ts)); });
  }, 15000);
}

// Recommended users (X layout, Discord presence)
export function userRowsHtml() {
  return [SELF, ...ALTS].map((account) => {
    const typing = (typingCounts.get(account.id) || 0) > 0;
    const status = typing ? "入力中…" : "オンライン";
    const on = following.has(account.id);
    return `<div class="user-row" data-typing="${typing}">
      ${avatar(account, "", true)}
      <div class="user-copy"><div class="user-name"><span>${escapeHtml(account.name)}</span>${badge(account)}</div><div class="user-status">@${account.handle} · ${status}</div></div>
      <button class="follow" type="button" data-follow="${account.id}" aria-pressed="${on}" aria-label="${on ? `${escapeHtml(account.name)}のフォローを解除` : `${escapeHtml(account.name)}をフォロー`}"><span class="on">フォロー中</span><span class="off">フォロー解除</span></button>
    </div>`;
  }).join("");
}

export function renderUsers() {
  $("users").innerHTML = userRowsHtml();
  const emptyList = document.querySelector("#empty .follow-list");
  if (emptyList) emptyList.innerHTML = userRowsHtml();
}
