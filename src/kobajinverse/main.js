// kobajinverse: wires the page together and starts the timeline.
//
//   llm.js, mock_llm.js, prompts.js   inference (one complete(task, input) call)
//   engine.js, model_picker.js        the WebLLM engine, start screen, and version
//   brain.js, safety.js               queue, cleanup, gates, and parsing
//   accounts.js, state.js, store.js   who is here, what this visit holds, saving it
//   dom.js, render.js                 HTML for posts, threads, and users
//   feed.js, notifications.js         views, filters, search, follows, 通知
//   actions.js                        reactions, replies, and kobajins talking
//   autopost.js, trends.js            the timeline's own life
//   composer.js, picker.js,
//   post_controls.js, profile.js      visitor input
//   share.js                          share cards (PNG)

import { setupComposer, focusComposer, insertText, syncComposer } from "./composer.js";
import { focusSearch, setFollow, setupFeed, setView, showPost, view } from "./feed.js";
import { catchUp, postLoop, rememberPosts, seedTimeline } from "./autopost.js";
import { modelReady, setupModelPicker } from "./model_picker.js";
import { onNotify, renderNotifications, unreadCount } from "./notifications.js";
import { setupPicker } from "./picker.js";
import { setupPostControls } from "./post_controls.js";
import { onProfileChange, setupProfile } from "./profile.js";
import { avatar, icon, refreshAll, renderUsers, startClock } from "./render.js";
import { posts } from "./state.js";
import { restoreTimeline } from "./store.js";
import { setupTrends, trendLoop } from "./trends.js";

// Buttons that appear in several places: trend tags, follow buttons, and
// the menu.
function setupNavigation() {
  document.addEventListener("click", (event) => {
    const tagButton = event.target.closest("[data-tag]");
    if (tagButton) { insertText(tagButton.dataset.tag); return; }
    const followButton = event.target.closest("[data-follow]");
    if (followButton) { setFollow(followButton.dataset.follow, followButton.getAttribute("aria-pressed") !== "true"); return; }
    const nav = event.target.closest("[data-nav]");
    if (!nav) return;
    if (nav.dataset.nav === "home") setView(view === "all" || view === "following" ? view : "all");
    if (nav.dataset.nav === "bookmarks") setView("bookmarks");
    if (nav.dataset.nav === "notifications") setView("notifications");
    if (nav.dataset.nav === "search") focusSearch();
    if (nav.dataset.nav === "compose") focusComposer();
  });
}

function paintBadges() {
  const count = unreadCount();
  document.querySelectorAll(".note-badge").forEach((el) => {
    el.hidden = !count || view === "notifications";
    el.textContent = count > 9 ? "9+" : String(count);
  });
  if (view === "notifications") renderNotifications(avatar, icon);
}

// A saved timeline comes back as it was; time away brings a few new
// posts. A first visit gets a fresh timeline.
async function startTimeline() {
  const saved = restoreTimeline();
  if (!saved.posts.length) { await seedTimeline(); return; }
  rememberPosts(saved.posts);
  [...saved.posts].reverse().forEach(showPost);
  await catchUp(saved.savedAt);
}

// Coming back to a tab left in the background for a while works the same.
function catchUpOnReturn() {
  let hiddenAt = 0;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { hiddenAt = Date.now(); return; }
    if (hiddenAt) catchUp(hiddenAt).catch(console.error);
    hiddenAt = 0;
  });
}

setupProfile();
setupFeed();
setupTrends();
setupComposer();
setupNavigation();
setupPicker();
setupPostControls();
setupModelPicker();
startClock();

onNotify(paintBadges);
onProfileChange(() => { refreshAll(posts); renderUsers(); });
paintBadges();
renderUsers();
refreshAll(posts);
syncComposer();
// The timeline is written by kobajinAI, so it starts once a version is
// running; until then the start screen covers the page (model_picker.js).
modelReady.then(() => {
  startTimeline().catch(console.error).finally(() => {
    catchUpOnReturn();
    postLoop();
  });
  trendLoop();
});

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register("/sw.js").catch(() => { /* the page works without it */ });
}
