import { ACCOUNTS } from "./accounts.js";

// What this visit holds. Follows are kept in localStorage; posts,
// reactions, and bookmarks last only for the visit.

export const posts = new Map();
export const pending = [];
export const typingCounts = new Map();
let postSeq = 0;

export const FOLLOW_KEY = "kobajinverse.following";
export const following = new Set();
try { JSON.parse(localStorage.getItem(FOLLOW_KEY) || "[]").forEach((id) => { if (ACCOUNTS[id]) following.add(id); }); } catch { /* start empty */ }

export function newPost(authorId, text, createdAt = Date.now()) {
  const post = { id: `p${++postSeq}`, authorId, text, createdAt, reactions: new Map(), reposts: new Set(), bookmarked: false, replies: [], drafts: [], typing: new Set() };
  posts.set(post.id, post);
  return post;
}
