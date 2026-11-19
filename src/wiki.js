// Fetch a short technical article excerpt from Japanese Wikipedia through the
// public MediaWiki API (CORS via origin=*). No key, no server. Wikipedia text is
// CC BY-SA, so every excerpt keeps its page URL for attribution.

const API_URL = "https://ja.wikipedia.org/w/api.php";
export const MAX_EXCERPT_CHARS = 1500;
const MIN_EXCERPT_CHARS = 120;
const TOP_RESULTS = 5;
export const WIKIPEDIA_LICENSE = "CC BY-SA 4.0";

// Field words close to what kobajin works with: Scratch, Web, 3D, video, LLMs.
export const TOPICS = [
  "Scratch (プログラミング言語)",
  "ビジュアルプログラミング言語",
  "JavaScript",
  "WebAssembly",
  "WebGL",
  "WebGPU",
  "HTML",
  "Cascading Style Sheets",
  "ウェブブラウザ",
  "静的ウェブサイト",
  "3次元コンピュータグラフィックス",
  "Blender",
  "レンダリング (コンピュータグラフィックス)",
  "シェーダ",
  "ポリゴン",
  "モーションキャプチャ",
  "ノンリニア編集",
  "動画共有サービス",
  "コーデック",
  "フレームレート",
  "大規模言語モデル",
  "トランスフォーマー (機械学習モデル)",
  "機械学習",
  "ニューラルネットワーク",
  "オープンソース",
  "ゲームエンジン",
  "コンピュータアニメーション",
  "ピクセルアート",
  "フォント",
  "GPU",
];

/** Collapse whitespace and remove the empty brackets explaintext leaves behind. */
export function cleanExtract(text) {
  return String(text || "")
    .replace(/\r/g, "")
    .replace(/[（(][\s、,;；:：]*[）)]/g, "")
    .replace(/[（(][^（）():：]*[:：]\s*[、,]?\s*[）)]/g, "")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

/** Keep at most `limit` characters, cutting at the last sentence end. */
export function truncateExcerpt(text, limit = MAX_EXCERPT_CHARS) {
  const chars = [...text];
  if (chars.length <= limit) return text;
  const head = chars.slice(0, limit).join("");
  const end = Math.max(head.lastIndexOf("。"), head.lastIndexOf("\n"));
  return (end > limit * 0.4 ? head.slice(0, end + 1) : head).trim();
}

function isUsable(page) {
  if (!page || page.missing || typeof page.extract !== "string") return false;
  if (/曖昧さ回避|一覧/.test(page.title)) return false;
  if (/曖昧さ回避|以下の(?:もの|意味|項目)を指す|次のような意味/.test(page.extract)) return false;
  return cleanExtract(page.extract).length >= MIN_EXCERPT_CHARS;
}

function toArticle(page, topic) {
  const url = page.canonicalurl || page.fullurl
    || `https://ja.wikipedia.org/wiki/${encodeURIComponent(page.title.replace(/ /g, "_"))}`;
  return {
    title: page.title,
    url,
    excerpt: truncateExcerpt(cleanExtract(page.extract)),
    pageid: page.pageid,
    revid: page.lastrevid ?? null,
    topic,
    license: WIKIPEDIA_LICENSE,
    fetchedAt: Date.now(),
  };
}

async function searchPages(topic, { fetchImpl, signal, headers }) {
  const params = new URLSearchParams({
    action: "query",
    format: "json",
    formatversion: "2",
    origin: "*",
    generator: "search",
    gsrsearch: topic,
    gsrnamespace: "0",
    gsrlimit: "20",
    prop: "extracts|info",
    exintro: "1",
    explaintext: "1",
    exlimit: "20",
    inprop: "url",
    redirects: "1",
  });
  const response = await fetchImpl(`${API_URL}?${params}`, { signal, headers });
  if (!response.ok) throw new Error(`Wikipedia API returned ${response.status}`);
  const data = await response.json();
  return data?.query?.pages || [];
}

/**
 * Pick one usable article for a random topic, skipping URLs in `exclude`.
 * Options: topic, exclude (Set of URLs), random, fetchImpl, signal, headers.
 */
export async function fetchArticle({
  topic,
  exclude = new Set(),
  random = Math.random,
  fetchImpl = globalThis.fetch.bind(globalThis),
  signal,
  headers,
} = {}) {
  const topics = topic ? [topic] : [...TOPICS].sort(() => random() - 0.5).slice(0, 3);
  for (const candidateTopic of topics) {
    // Search order is relevance order; lower-ranked hits drift off-topic.
    const pages = (await searchPages(candidateTopic, { fetchImpl, signal, headers }))
      .sort((left, right) => (left.index ?? 99) - (right.index ?? 99))
      .slice(0, TOP_RESULTS)
      .filter(isUsable)
      .map((page) => toArticle(page, candidateTopic))
      .filter((article) => !exclude.has(article.url));
    if (pages.length > 0) return pages[Math.floor(random() * pages.length)];
  }
  throw new Error("No usable Wikipedia article was found for the chosen topics");
}
