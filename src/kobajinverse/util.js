// Small helpers shared by the timeline modules.

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
export const rand = (min, max) => min + Math.random() * (max - min);
export const randInt = (min, max) => Math.floor(rand(min, max + 1));
export const pick = (list) => list[Math.floor(Math.random() * list.length)];
export const shuffle = (list) => list.map((item) => [Math.random(), item]).sort((a, b) => a[0] - b[0]).map(([, item]) => item);

export const chars = (text) => Array.from(text);

// Background work waits while the tab is hidden.
export async function whenVisible() {
  while (document.hidden) await sleep(1000);
}
