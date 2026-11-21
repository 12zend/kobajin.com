import { ACCOUNTS } from "./accounts.js";
import { $, escapeHtml } from "./dom.js";

// Notifications: what the kobajins did to the visitor's posts and
// replies. Reactions with the same emoji on the same post fold into one
// line while unread. Kept in this browser.

const KEY = "kobajinverse.notifications.v1";
const MAX = 60;
let notes = [];
let seq = 0;
const listeners = new Set();

try {
  const saved = JSON.parse(localStorage.getItem(KEY) || "[]");
  if (Array.isArray(saved)) {
    notes = saved.filter((note) => note && (note.actors || []).every((id) => ACCOUNTS[id])).slice(0, MAX);
    seq = notes.reduce((max, note) => Math.max(max, Number(String(note.id).slice(1)) || 0), 0);
  }
} catch { /* start empty */ }

function save() {
  try { localStorage.setItem(KEY, JSON.stringify(notes)); } catch { /* this visit only */ }
}

export const unreadCount = () => notes.filter((note) => !note.read).length;
export const onNotify = (listener) => listeners.add(listener);
const changed = () => { save(); listeners.forEach((listener) => listener()); };

export function notify({ type, post, actorId, emoji = "", text = "" }) {
  const top = notes.find((note) => !note.read && note.type === type && note.postId === post.id && note.emoji === emoji && note.text === text);
  if (top && type !== "reply") {
    if (!top.actors.includes(actorId)) top.actors.unshift(actorId);
    top.at = Date.now();
    notes = [top, ...notes.filter((note) => note !== top)];
  } else {
    notes.unshift({ id: `n${++seq}`, type, postId: post.id, postText: post.text, actors: [actorId], emoji, text, at: Date.now(), read: false });
  }
  notes = notes.slice(0, MAX);
  changed();
}

export function markAllRead() {
  if (!notes.some((note) => !note.read)) return;
  notes.forEach((note) => { note.read = true; });
  changed();
}

function who(actors) {
  const names = actors.map((id) => `<b>${escapeHtml(ACCOUNTS[id].name)}</b>`);
  return names.length > 2 ? `${names.slice(0, 2).join("、")} ほか${names.length - 2}人` : names.join("、");
}

function line(note) {
  if (note.type === "reaction") return `${who(note.actors)} が${note.text ? "あなたの返信に" : ""} ${note.emoji} でリアクションしました`;
  if (note.type === "repost") return `${who(note.actors)} がリポストしました`;
  return `${who(note.actors)} が返信しました`;
}

const ICON = { reaction: "react", repost: "repost", reply: "reply" };

export function renderNotifications(avatarHtml, iconHtml) {
  const list = $("notes");
  if (!notes.length) {
    list.innerHTML = '<div class="empty"><h2>まだ通知はありません</h2><p>ポストすると、kobajinたちのリアクションや返信がここに届きます。</p></div>';
    return;
  }
  list.innerHTML = notes.map((note) => `<button class="note${note.read ? "" : " unread"}" type="button" data-note-post="${escapeHtml(note.postId)}">
      <span class="note-icon ${note.type}">${note.type === "reaction" ? `<span class="note-emoji">${note.emoji}</span>` : iconHtml(ICON[note.type])}</span>
      <span class="note-body">
        <span class="note-faces">${note.actors.slice(0, 5).map((id) => avatarHtml(ACCOUNTS[id], "xs")).join("")}</span>
        <span class="note-line">${line(note)}</span>
        ${note.text ? `<span class="note-reply">${escapeHtml(note.text)}</span>` : ""}
        <span class="note-post">${escapeHtml(note.postText)}</span>
      </span>
    </button>`).join("");
}
