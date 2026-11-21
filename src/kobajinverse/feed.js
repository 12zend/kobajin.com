import { ACCOUNTS, VISITOR } from "./accounts.js";
import { $, escapeHtml, toast } from "./dom.js";
import { markAllRead, renderNotifications } from "./notifications.js";
import { avatar, feedEl, icon, postHtml, renderUsers, userRowsHtml } from "./render.js";
import { FOLLOW_KEY, following, pending, posts } from "./state.js";
import { scheduleSave } from "./store.js";

// Views and filters: おすすめ, フォロー中, ブックマーク, 通知, search. New posts
// enter the feed through showPost() and arrive().

export let view = "all";

const searchInputs = [...document.querySelectorAll(".search-input")];
let query = "";

function postElement(post) {
  const el = document.createElement("article");
  el.className = "post enter";
  el.dataset.id = post.id;
  el.innerHTML = postHtml(post);
  el.hidden = !isVisible(post);
  return el;
}

export function showPost(post) {
  feedEl.prepend(postElement(post));
  applyFilters();
  scheduleSave();
}

// New posts wait behind a pill while the reader is scrolled down, so the
// timeline never jumps under them.
export function arrive(post) {
  if (window.scrollY > 240) {
    pending.push(post);
    const faces = [...new Set(pending.map((p) => p.authorId))].slice(0, 3).map((id) => avatar(ACCOUNTS[id], "sm")).join("");
    $("new-posts-button").innerHTML = `${icon("up")}<span style="display:flex">${faces}</span>がポストしました`;
    $("new-posts-button").querySelectorAll(".avatar").forEach((el, index) => { el.style.cssText = `width:24px;height:24px;margin-left:${index ? -8 : 0}px;border:2px solid var(--blue)`; });
    $("new-posts").hidden = false;
  } else {
    showPost(post);
  }
}

function matchesSearch(post) {
  if (!query) return true;
  const account = ACCOUNTS[post.authorId];
  return [post.text, account.name, account.handle, ...post.replies.map((reply) => reply.text)].join(" ").toLowerCase().includes(query);
}

function inView(post) {
  if (view === "following") return following.has(post.authorId) || post.authorId === VISITOR.id;
  if (view === "bookmarks") return post.bookmarked;
  return true;
}

const isVisible = (post) => inView(post) && matchesSearch(post);

export function applyFilters() {
  if (view === "notifications") { $("empty").innerHTML = ""; return; }
  let shown = 0;
  feedEl.querySelectorAll(".post").forEach((el) => {
    const visible = isVisible(posts.get(el.dataset.id));
    el.hidden = !visible;
    if (visible) shown += 1;
  });
  const empty = $("empty");
  if (shown || feedEl.querySelector(".skeleton")) { empty.innerHTML = ""; return; }
  if (query) {
    empty.innerHTML = `<div class="empty"><h2>「${escapeHtml(query)}」の検索結果はありません</h2><p>別のキーワードを試してみてください。</p></div>`;
  } else if (view === "following") {
    empty.innerHTML = `<div class="empty"><h2>まだ誰もフォローしていません</h2><p>kobajinをフォローすると、そのポストがここに並びます。</p><div class="follow-list">${userRowsHtml()}</div></div>`;
  } else if (view === "bookmarks") {
    empty.innerHTML = '<div class="empty"><h2>ブックマークはまだありません</h2><p>気になったポストをブックマークすると、ここに表示されます。</p></div>';
  } else {
    empty.innerHTML = '<div class="empty"><p>まだ誰もポストしていません</p></div>';
  }
}

const TITLES = { bookmarks: "ブックマーク", notifications: "通知" };

export function setView(next) {
  view = next;
  const titled = Boolean(TITLES[view]);
  document.querySelectorAll(".tab").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.view === view)));
  $("tabs").hidden = titled;
  $("view-title").hidden = !titled;
  $("view-title-text").textContent = TITLES[view] || "";
  $("composer").hidden = titled;
  $("trend-strip").hidden = titled;
  feedEl.hidden = view === "notifications";
  $("notes").hidden = view !== "notifications";
  if (view === "notifications") {
    renderNotifications(avatar, icon);
    markAllRead();
  }
  const navFor = (v) => (TITLES[v] ? v : "home");
  document.querySelectorAll(".nav-item[data-nav], .bottom-bar [data-nav]").forEach((el) => {
    if (el.dataset.nav === navFor(view)) el.setAttribute("aria-current", "page"); else el.removeAttribute("aria-current");
  });
  applyFilters();
  window.scrollTo({ top: 0 });
}

// Opens a post from a notification: back to the timeline, scrolled to it.
export function showPostById(id) {
  setView("all");
  const el = feedEl.querySelector(`[data-id="${CSS.escape(id)}"]`);
  if (!el) return false;
  el.scrollIntoView({ block: "center" });
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
  return true;
}

export function setFollow(id, on) {
  if (on) following.add(id); else following.delete(id);
  try { localStorage.setItem(FOLLOW_KEY, JSON.stringify([...following])); } catch { /* this visit only */ }
  renderUsers();
  applyFilters();
}

export function focusSearch() {
  const visible = searchInputs.find((el) => el.offsetParent !== null);
  visible?.focus();
}

export function setupFeed() {
  $("notes").addEventListener("click", (event) => {
    const note = event.target.closest("[data-note-post]");
    if (note && !showPostById(note.dataset.notePost)) toast("このポストはもう表示されていません");
  });

  $("new-posts-button").addEventListener("click", () => {
    pending.splice(0).forEach(showPost);
    $("new-posts").hidden = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  $("tabs").addEventListener("click", (event) => {
    const tab = event.target.closest("[data-view]");
    if (tab) setView(tab.dataset.view);
  });

  searchInputs.forEach((inputEl) => {
    inputEl.addEventListener("input", () => {
      query = inputEl.value.trim().toLowerCase();
      searchInputs.forEach((other) => { if (other !== inputEl) other.value = inputEl.value; });
      applyFilters();
    });
  });
}
