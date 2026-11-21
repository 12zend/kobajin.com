import { VISITOR } from "./accounts.js";
import { choreograph } from "./actions.js";
import { $, toast } from "./dom.js";
import { setView, showPost, view } from "./feed.js";
import { wakeModel } from "./model_picker.js";
import { openPicker } from "./picker.js";
import { newPost } from "./state.js";
import { trendTags } from "./trends.js";
import { chars, pick } from "./util.js";

// Composer: the visitor's post box and its tools.

const composer = $("composer");
const input = $("composer-text");
const postButton = $("post-button");
export const LIMIT = 140;
const RING = 69.1;

export function syncComposer() {
  // An empty box keeps its CSS height; measuring it before layout settles
  // can return the height of a much narrower box.
  input.style.height = "";
  if (input.value) {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
  }
  const used = chars(input.value).length;
  const left = LIMIT - used;
  const ring = $("ring");
  ring.hidden = !used;
  ring.dataset.state = left < 0 ? "over" : left <= 20 ? "near" : "ok";
  $("ring-bar").setAttribute("stroke-dashoffset", String(RING * (1 - Math.min(used, LIMIT) / LIMIT)));
  $("ring-left").textContent = left <= 20 ? left : "";
  postButton.disabled = !input.value.trim() || left < 0;
}

export function insertText(text, { space = true } = {}) {
  const start = input.selectionStart ?? input.value.length;
  const end = input.selectionEnd ?? input.value.length;
  const before = input.value.slice(0, start);
  const sep = space && before && !/\s$/.test(before) ? " " : "";
  const piece = `${sep}${text}${space ? " " : ""}`;
  input.value = `${before}${piece}${input.value.slice(end)}`;
  const caret = before.length + piece.length;
  syncComposer();
  input.focus();
  input.setSelectionRange(caret, caret);
}

const onTimeline = () => view === "all" || view === "following";

export function focusComposer() {
  if (!onTimeline()) setView("all");
  window.scrollTo({ top: 0, behavior: "smooth" });
  input.focus({ preventScroll: true });
}

export function setupComposer() {
  input.addEventListener("input", syncComposer);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.isComposing) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });

  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || chars(text).length > LIMIT) return;
    input.value = "";
    syncComposer();
    if (!onTimeline()) setView("all");
    wakeModel();
    const post = newPost(VISITOR.id, text);
    showPost(post);
    window.scrollTo({ top: 0, behavior: "smooth" });
    toast("ポストを送信しました");
    choreograph(post);
  });

  $("emoji-tool").addEventListener("click", (event) => {
    openPicker(event.currentTarget, (emoji) => insertText(emoji, { space: false }));
  });
  $("tag-tool").addEventListener("click", () => {
    if (trendTags.length) insertText(pick(trendTags).tag); else insertText("#", { space: false });
  });
}
