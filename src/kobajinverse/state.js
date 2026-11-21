import { ACCOUNTS } from "./accounts.js";

// What this visit holds. Follows are kept in localStorage; the timeline
// (posts, reactions, replies, bookmarks) is saved by store.js.

export const posts = new Map();
export const pending = [];
export const typingCounts = new Map();
let postSeq = 0;

export const FOLLOW_KEY = "kobajinverse.following";
export const following = new Set();
try { JSON.parse(localStorage.getItem(FOLLOW_KEY) || "[]").forEach((id) => { if (ACCOUNTS[id]) following.add(id); }); } catch { /* start empty */ }

// `source` is what wrote the text: "mock", a kobajinAI version ("v2"),
// or "" for the visitor.
export function newPost(authorId, text, createdAt = Date.now(), source = "") {
  const post = { id: `p${++postSeq}`, authorId, text, createdAt, source, reactions: new Map(), reposts: new Set(), bookmarked: false, replies: [], drafts: [], typing: new Set() };
  posts.set(post.id, post);
  return post;
}

// Takes in a post saved on an earlier visit (store.js).
export function adoptPost(post) {
  posts.set(post.id, post);
  postSeq = Math.max(postSeq, Number(post.id.slice(1)) || 0);
}
