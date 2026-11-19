// Rule gates for kobajinAI feed posts. Pure functions with no browser or Vite
// dependencies, so self_twin/evaluation can run the same checks under Node.

// ---------------------------------------------------------------------------
// Text helpers

/** Strip think blocks and wrapping quotes/labels that the model sometimes adds. */
export function cleanModelText(output) {
  let text = String(output || "")
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>[\s\S]*$/gi, "")
    .replace(/<\/?think>/gi, "")
    .trim();
  text = text.replace(/^```(?:json)?\s*\n?([\s\S]*?)\n?```$/i, "$1").trim();
  text = text.replace(/^(?:呟き|投稿|返信|本文)\s*[:：]\s*/, "").trim();
  const quoted = /^[「『"“](.*)[」』"”]$/s.exec(text);
  if (quoted && !/[「『"“」』"”]/.test(quoted[1])) text = quoted[1].trim();
  return text;
}

/** Strict JSON parse after removing whitespace. Returns null on any failure. */
export function parseJsonOutput(text) {
  try {
    return JSON.parse(String(text || "").trim());
  } catch {
    return null;
  }
}

/**
 * Conservative token estimate for Qwen-family tokenizers: CJK and other
 * non-ASCII characters count as one token each, ASCII runs as ~3 chars/token.
 * It overestimates Japanese slightly, which keeps requests inside the window.
 */
export function estimateTokens(text) {
  let nonAscii = 0;
  let ascii = 0;
  for (const char of String(text || "")) {
    if (char.codePointAt(0) > 0x7f) nonAscii += 1;
    else ascii += 1;
  }
  return nonAscii + Math.ceil(ascii / 3);
}

/** Cut text at a sentence boundary so it fits the token budget. */
export function trimToTokenBudget(text, maxTokens) {
  const source = String(text || "");
  if (maxTokens <= 0) return "";
  if (estimateTokens(source) <= maxTokens) return source;
  const sentences = source.match(/[^。！？!?\n]+[。！？!?\n]*/g) || [source];
  let result = "";
  for (const sentence of sentences) {
    if (estimateTokens(result + sentence) > maxTokens) break;
    result += sentence;
  }
  if (result) return result.trim();
  // A single very long sentence: hard cut by characters.
  let cut = "";
  for (const char of source) {
    if (estimateTokens(cut + char) > maxTokens) break;
    cut += char;
  }
  return cut;
}

export function normalizeForMatch(text) {
  return String(text || "").normalize("NFKC").toLocaleLowerCase("ja-JP").replace(/\s+/g, "");
}

function normalizeForSimilarity(text) {
  return normalizeForMatch(text).replace(/[、。，．,.!！?？「」『』（）()【】・…ー〜~"'“”:：;；\-]/g, "");
}

export function charBigrams(text) {
  const chars = [...normalizeForSimilarity(text)];
  const grams = new Set();
  for (let index = 0; index < chars.length - 1; index += 1) grams.add(chars[index] + chars[index + 1]);
  return grams;
}

export function bigramJaccard(left, right) {
  const a = charBigrams(left);
  const b = charBigrams(right);
  if (a.size === 0 && b.size === 0) return 0;
  let shared = 0;
  for (const gram of a) if (b.has(gram)) shared += 1;
  return shared / (a.size + b.size - shared);
}

/** Share of the reply's bigrams that also occur in the topic text (0–1). */
export function topicOverlap(reply, topic) {
  const a = charBigrams(reply);
  if (a.size === 0) return 0;
  const b = charBigrams(topic);
  let shared = 0;
  for (const gram of a) if (b.has(gram)) shared += 1;
  return shared / a.size;
}

/** Length of the longest substring of `text` that also appears in `source`. */
export function longestCommonSubstring(text, source) {
  const a = [...normalizeForMatch(text)];
  const b = [...normalizeForMatch(source)];
  if (a.length === 0 || b.length === 0) return 0;
  let previous = new Uint16Array(b.length + 1);
  let current = new Uint16Array(b.length + 1);
  let best = 0;
  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      current[j] = a[i - 1] === b[j - 1] ? previous[j - 1] + 1 : 0;
      if (current[j] > best) best = current[j];
    }
    [previous, current] = [current, previous];
    current.fill(0);
  }
  return best;
}

// ---------------------------------------------------------------------------
// Form gate

export const MAX_TWEET_CHARS = 140;
const EMOJI = /\p{Extended_Pictographic}/u;
const URL_LIKE = /https?:\/\/|www\.|\.(?:com|org|jp|net|io)\b/i;
const HASHTAG = /[#＃][^\s#＃]/;

export function formGate(text, finishReason = null) {
  const value = String(text || "").trim();
  if (!value) return { pass: false, reason: "empty" };
  if (finishReason === "length") return { pass: false, reason: "truncated" };
  if ([...value].length > MAX_TWEET_CHARS) return { pass: false, reason: "too_long" };
  if (HASHTAG.test(value)) return { pass: false, reason: "hashtag" };
  if (URL_LIKE.test(value)) return { pass: false, reason: "url" };
  if (EMOJI.test(value)) return { pass: false, reason: "emoji" };
  // A post is one voice; standalone quoted lines are leaked dialogue turns.
  if (value.split("\n").some((line) => /^[「『][^」』]*[」』]$/.test(line.trim()))) return { pass: false, reason: "dialogue" };
  return { pass: true };
}

// ---------------------------------------------------------------------------
// Fact gate (rule part)

// Casual katakana and laughter that carry no factual claim.
const CASUAL_TERMS = new Set([
  "マジ", "ヤバ", "ヤバい", "ガチ", "ワクワク", "ドキドキ", "スゴ", "エグ", "ムズ", "ホント", "チョー",
  "イイ", "ナイス", "オモロ", "ウケ", "ネタ", "ノリ", "ピンと", "ピン", "テンション", "レベル", "イメージ",
  "w", "ww", "www", "lol", "ok",
]);

/** Numbers, years, Latin-script names, and katakana words that read as factual claims. */
export function extractClaimTerms(text) {
  const value = String(text || "").normalize("NFKC");
  const terms = new Set();
  for (const match of value.matchAll(/\d+(?:[.,]\d+)*/g)) terms.add(match[0]);
  for (const match of value.matchAll(/[A-Za-z][A-Za-z0-9+#]*(?:[.\-'][A-Za-z0-9+#]+)*/g)) terms.add(match[0]);
  for (const match of value.matchAll(/[ァ-ヺヽヾ][ァ-ヺーヽヾ]+/g)) terms.add(match[0]);
  return [...terms].filter((term) => !CASUAL_TERMS.has(term.toLowerCase()) && !CASUAL_TERMS.has(term));
}

function termIsSupported(term, haystack) {
  const needle = normalizeForMatch(term);
  if (haystack.includes(needle)) return true;
  // Allow a trailing long-vowel mark difference (サーバ / サーバー).
  if (needle.endsWith("ー") && haystack.includes(needle.slice(0, -1))) return true;
  return haystack.includes(`${needle}ー`);
}

export function factRuleGate(text, excerpt, title = "") {
  const haystack = normalizeForMatch(`${title}\n${excerpt}`);
  const missing = extractClaimTerms(text).filter((term) => !termIsSupported(term, haystack));
  return { pass: missing.length === 0, missing };
}

/** Keep only seed facts whose claim terms appear in the excerpt. */
export function filterSupportedFacts(facts, excerpt, title = "") {
  return (facts || []).filter((fact) => factRuleGate(fact, excerpt, title).pass);
}

// ---------------------------------------------------------------------------
// Identity gate: Wikipedia content must not become kobajin's own experience.

const EXPERIENCE_PATTERNS = [
  /昔|むかし|以前|子(?:供|ども)の頃|小さい頃|学生(?:の頃|時代)|若い頃|当時の(?:僕|俺|私|自分)/,
  /懐かし|思い出[すしした]/,
  /(?:使|つか|触|さわ|や|遊|ハマ|はま|習|かよ|通|持|作|見|読|聴|聞|行)(?:っ|ん|い)?[てで](?:た|ました|いた)/,
  /(?:使|触|や|遊|習|作|読|行|買|書)(?:っ|い)?た(?:こと|事)(?:が)?ある/,
  /(?:僕|俺|私|わたし|自分)(?:も|は|が)?(?:よく|ずっと|前から|昔から|普段|毎日|いつも)/,
  /(?:僕|俺|私|わたし|自分)の(?:家|部屋|PC|パソコン|学校|会社|友達|地元|親|職場)/,
  /愛用|大好き|ずっと好き|昔から好き|前から好き|推してる/,
  /(?:僕|俺|私|わたし|自分)も(?:[^。！？\n]{0,10})(?:使|作|や|触|遊|習|買|行)/,
];

export function identityGate(text, memoryRecords = []) {
  const value = String(text || "").normalize("NFKC");
  const hits = [];
  for (const pattern of EXPERIENCE_PATTERNS) {
    const match = pattern.exec(value);
    if (match) hits.push(match[0]);
  }
  if (hits.length === 0) return { pass: true, hits, backed: false };
  // First-person experience is allowed only when a confirmed memory matches.
  const backed = Array.isArray(memoryRecords) && memoryRecords.length > 0;
  return { pass: backed, hits, backed };
}

// ---------------------------------------------------------------------------
// Novelty gate

export const NOVELTY_WINDOW = 50;
export const MAX_JACCARD = 0.5;
export const MAX_COPY_CHARS = 20;

export function noveltyGate(text, recentTexts = [], excerpt = "") {
  let maxJaccard = 0;
  for (const previous of recentTexts.slice(0, NOVELTY_WINDOW)) {
    maxJaccard = Math.max(maxJaccard, bigramJaccard(text, previous));
  }
  const copyLength = longestCommonSubstring(text, excerpt);
  return {
    pass: maxJaccard < MAX_JACCARD && copyLength < MAX_COPY_CHARS,
    maxJaccard,
    copyLength,
  };
}
