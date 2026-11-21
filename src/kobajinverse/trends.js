import { brain } from "./brain.js";
import { llm } from "./llm.js";
import { $, escapeHtml } from "./dom.js";
import { rand, randInt, sleep, whenVisible } from "./util.js";

// Trends: いまどうしてる？ in the sidebar and the chip strip.

export let trendTags = [];
let trendsExpanded = false;

function renderTrends() {
  const shown = trendsExpanded ? trendTags : trendTags.slice(0, 5);
  $("trends").innerHTML = shown.map((item, index) => `<button class="trend enter" type="button" data-tag="${escapeHtml(item.tag)}">
      <div class="trend-cat">${index + 1}位 · kobajin界隈のトレンド</div>
      <div class="trend-tag">${escapeHtml(item.tag)}</div>
      <div class="trend-count">${item.count}件のポスト</div>
    </button>`).join("");
  $("trends-more").hidden = trendTags.length <= 5;
  $("trends-more").textContent = trendsExpanded ? "表示を減らす" : "さらに表示";
  $("trend-strip").innerHTML = `<span class="strip-label">いまどうしてる？</span>${trendTags.map((item) => `<button type="button" data-tag="${escapeHtml(item.tag)}">${escapeHtml(item.tag)}</button>`).join("")}`;
}

export function setupTrends() {
  $("trends-more").addEventListener("click", () => { trendsExpanded = !trendsExpanded; renderTrends(); });
}

export async function trendLoop() {
  const format = (n) => (n >= 10000 ? `${(n / 10000).toFixed(1)}万` : n.toLocaleString("ja-JP"));
  for (;;) {
    await whenVisible();
    try {
      const tags = await brain.trends(10);
      if (tags.length) {
        const counts = tags.map(() => randInt(800, 48000)).sort((a, b) => b - a);
        trendTags = tags.map((tag, index) => ({ tag, count: format(counts[index]) }));
        renderTrends();
      }
    } catch (error) { console.error(error); }
    // kobajinAI writes 120 tokens per refresh, so it refreshes less often.
    await sleep(llm.name === "mock" ? rand(80000, 110000) : rand(240000, 330000));
  }
}
