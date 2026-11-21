import { VISITOR } from "./accounts.js";
import { $, escapeHtml, toast } from "./dom.js";
import { chars } from "./util.js";

// The visitor's name and avatar (an emoji on a color), kept in this
// browser. Nothing about the visitor is sent anywhere.

const KEY = "kobajinverse.profile";
export const AVATAR_EMOJI = ["🐱", "🐶", "🐸", "🐧", "🦊", "🐼", "🍙", "🍣", "🌙", "⭐", "🎧", "🌱"];
export const AVATAR_COLORS = ["#1d9bf0", "#7856ff", "#f91880", "#ff7a00", "#00ba7c", "#ffd400", "#536471"];
const NAME_MAX = 20;
const listeners = new Set();

function apply(profile) {
  const name = chars(String(profile?.name || "").trim()).slice(0, NAME_MAX).join("");
  VISITOR.name = name || "あなた";
  VISITOR.emoji = AVATAR_EMOJI.includes(profile?.emoji) ? profile.emoji : "";
  VISITOR.color = AVATAR_COLORS.includes(profile?.color) ? profile.color : "";
}

try { apply(JSON.parse(localStorage.getItem(KEY) || "null")); } catch { /* defaults */ }

export const onProfileChange = (listener) => listeners.add(listener);

function setProfile(profile) {
  apply(profile);
  try { localStorage.setItem(KEY, JSON.stringify({ name: VISITOR.name === "あなた" ? "" : VISITOR.name, emoji: VISITOR.emoji, color: VISITOR.color })); } catch { /* this visit only */ }
  listeners.forEach((listener) => listener());
}

export function setupProfile() {
  const dialog = $("profile-dialog");
  const form = $("profile-form");
  const nameInput = $("profile-name");
  const preview = $("profile-preview");
  let emoji = VISITOR.emoji;
  let color = VISITOR.color;

  const paintPreview = () => {
    preview.textContent = emoji;
    preview.style.background = color || "";
    preview.classList.toggle("has-emoji", Boolean(emoji));
    form.querySelectorAll("[data-emoji-choice]").forEach((el) => el.setAttribute("aria-pressed", String(el.dataset.emojiChoice === emoji)));
    form.querySelectorAll("[data-color-choice]").forEach((el) => el.setAttribute("aria-pressed", String(el.dataset.colorChoice === color)));
  };

  $("profile-emoji").innerHTML = AVATAR_EMOJI.map((choice) => `<button type="button" class="choice" data-emoji-choice="${choice}" aria-label="${choice}">${choice}</button>`).join("");
  $("profile-color").innerHTML = AVATAR_COLORS.map((choice) => `<button type="button" class="swatch" data-color-choice="${choice}" style="background:${choice}" aria-label="色 ${choice}"></button>`).join("");

  form.addEventListener("click", (event) => {
    const pickEmoji = event.target.closest("[data-emoji-choice]");
    if (pickEmoji) { emoji = pickEmoji.dataset.emojiChoice === emoji ? "" : pickEmoji.dataset.emojiChoice; paintPreview(); }
    const pickColor = event.target.closest("[data-color-choice]");
    if (pickColor) { color = pickColor.dataset.colorChoice; paintPreview(); }
    if (event.target.closest("[data-close]")) dialog.close();
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    setProfile({ name: nameInput.value, emoji, color });
    dialog.close();
    toast("プロフィールを保存しました");
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest('[data-nav="profile"]')) return;
    emoji = VISITOR.emoji;
    color = VISITOR.color;
    nameInput.value = VISITOR.name === "あなた" ? "" : VISITOR.name;
    nameInput.maxLength = NAME_MAX;
    paintPreview();
    dialog.showModal();
  });

  const paintChip = () => { document.querySelectorAll(".me-name").forEach((el) => { el.innerHTML = escapeHtml(VISITOR.name); }); };
  onProfileChange(paintChip);
  paintChip();
}
