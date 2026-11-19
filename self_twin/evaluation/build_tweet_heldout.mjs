#!/usr/bin/env node
// Build the fixed 20-article held-out set for feed-post evaluation, using the
// same fetch and excerpt code as the website (src/wiki.js). Wikipedia text is
// CC BY-SA 4.0; every row keeps its page URL and revision for attribution.
//
//   node self_twin/evaluation/build_tweet_heldout.mjs [output.jsonl]

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { fetchArticle } from "../../src/wiki.js";

const HELDOUT_TOPICS = [
  "Scratch (プログラミング言語)",
  "ビジュアルプログラミング言語",
  "JavaScript",
  "WebAssembly",
  "WebGL",
  "HTML",
  "Cascading Style Sheets",
  "ウェブブラウザ",
  "3次元コンピュータグラフィックス",
  "Blender",
  "シェーダ",
  "モーションキャプチャ",
  "ノンリニア編集",
  "コーデック",
  "フレームレート",
  "大規模言語モデル",
  "トランスフォーマー (機械学習モデル)",
  "ニューラルネットワーク",
  "ゲームエンジン",
  "ピクセルアート",
];
const HEADERS = { "User-Agent": "kobajinAI-feed-eval/0.1 (offline evaluation script)" };
const output = process.argv[2]
  || fileURLToPath(new URL("./tweet_heldout_articles.jsonl", import.meta.url));

const pause = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

const rows = [];
const seen = new Set();
for (const topic of HELDOUT_TOPICS) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      // random() = 0 picks the top-ranked usable result, so the set is reproducible.
      const article = await fetchArticle({ topic, exclude: seen, random: () => 0, headers: HEADERS });
      seen.add(article.url);
      const { fetchedAt, ...stable } = article;
      rows.push({ index: rows.length, ...stable, fetched_at: new Date(fetchedAt).toISOString() });
      console.error(`${rows.length}/${HELDOUT_TOPICS.length} ${topic} → ${article.title}`);
      break;
    } catch (error) {
      if (attempt >= 4 || !/429|5\d\d/.test(String(error.message))) throw error;
      await pause(5000 * 2 ** attempt);
    }
  }
  await pause(2000);
}
writeFileSync(output, rows.map((row) => JSON.stringify(row)).join("\n") + "\n");
console.error(`Wrote ${rows.length} articles to ${output}`);
