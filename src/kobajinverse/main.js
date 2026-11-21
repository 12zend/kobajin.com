// kobajinverse: wires the page together and starts the timeline.
//
//   llm.js, mock_llm.js, prompts.js   inference (one complete(task, input) call)
//   brain.js                          queue, cleanup, and parsing around the model
//   accounts.js, state.js             who is here and what this visit holds
//   dom.js, render.js                 HTML for posts, threads, and users
//   feed.js                           views, filters, search, follows
//   actions.js                        reactions, replies, and kobajins answering
//   autopost.js, trends.js            the timeline's own life
//   composer.js, picker.js,
//   post_controls.js, model_picker.js visitor input

import { setupComposer, focusComposer, insertText, syncComposer } from "./composer.js";
import { focusSearch, setFollow, setupFeed, setView, view } from "./feed.js";
import { postLoop, seedTimeline } from "./autopost.js";
import { setupModelPicker } from "./model_picker.js";
import { setupPicker } from "./picker.js";
import { setupPostControls } from "./post_controls.js";
import { renderUsers, startClock } from "./render.js";
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
    if (nav.dataset.nav === "home") setView(view === "bookmarks" ? "all" : view);
    if (nav.dataset.nav === "bookmarks") setView("bookmarks");
    if (nav.dataset.nav === "search") focusSearch();
    if (nav.dataset.nav === "compose") focusComposer();
  });
}

setupFeed();
setupTrends();
setupComposer();
setupNavigation();
setupPicker();
setupPostControls();
setupModelPicker();
startClock();

renderUsers();
syncComposer();
seedTimeline().then(postLoop);
trendLoop();

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register("/sw.js").catch(() => { /* the page works without it */ });
}
