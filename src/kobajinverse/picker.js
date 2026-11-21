// Emoji picker: reacts on a post, or types into the composer.
import { REACTION_EMOJI as PICKER_EMOJI } from "./prompts.js";
let picker = null;

function closePicker() { picker?.remove(); picker = null; }

export function openPicker(anchor, onPick) {
  closePicker();
  picker = document.createElement("div");
  picker.className = "picker";
  picker.setAttribute("role", "menu");
  picker.innerHTML = PICKER_EMOJI.map((emoji) => `<button type="button" role="menuitem" data-pick="${emoji}" aria-label="${emoji}">${emoji}</button>`).join("");
  document.body.append(picker);
  const box = anchor.getBoundingClientRect();
  picker.style.left = `${Math.max(8, Math.min(window.innerWidth - picker.offsetWidth - 8, box.left))}px`;
  picker.style.top = `${box.top > 70 ? box.top - picker.offsetHeight - 6 : box.bottom + 6}px`;
  picker.addEventListener("click", (event) => {
    const choice = event.target.closest("[data-pick]");
    if (!choice) return;
    onPick(choice.dataset.pick);
    closePicker();
  });
  picker.querySelector("button").focus();
}

// A small menu of actions under a button: [{ label, run }].
export function openMenu(anchor, items) {
  closePicker();
  picker = document.createElement("div");
  picker.className = "picker menu";
  picker.setAttribute("role", "menu");
  picker.innerHTML = items.map((item, index) => `<button type="button" role="menuitem" data-item="${index}">${item.label}</button>`).join("");
  document.body.append(picker);
  const box = anchor.getBoundingClientRect();
  picker.style.left = `${Math.max(8, Math.min(window.innerWidth - picker.offsetWidth - 8, box.right - picker.offsetWidth))}px`;
  picker.style.top = `${box.bottom + picker.offsetHeight + 6 < window.innerHeight ? box.bottom + 6 : box.top - picker.offsetHeight - 6}px`;
  picker.addEventListener("click", (event) => {
    const choice = event.target.closest("[data-item]");
    if (!choice) return;
    closePicker();
    items[Number(choice.dataset.item)].run();
  });
  picker.querySelector("button").focus();
}

export function setupPicker() {
  document.addEventListener("pointerdown", (event) => {
    if (picker && !picker.contains(event.target) && !event.target.closest('[data-add], #emoji-tool, [data-kind="share"]')) closePicker();
  });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closePicker(); });
  window.addEventListener("scroll", closePicker, { passive: true });
}
