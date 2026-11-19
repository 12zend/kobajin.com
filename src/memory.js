import publicMemories from "./public_memories.json";

const records = publicMemories
  .filter((record) => record.subject === "self" && record.review_status === "confirmed");

const searchTerms = {
  "self-20241125-minyo": ["民謡", "歌", "習った", "習ってた", "習っていた", "習い事"],
  "self-20240503-textblur": ["textblur", "作った", "作成", "プロジェクト", "保守", "使って"],
  "self-20240828-nanachiki": ["ナナチキ", "チキン", "食べ物", "食べるもの", "好きな食べ物"],
};

const MEMORY_INTENT = /前に|以前|昔|過去|覚えて|話した|言ってた|なんだっけ|何て言|いつ|どこ|好き|嫌い|苦手|興味|経験|自分で作|作った|プロジェクト|食べ物|スポーツ/i;

function normalize(value) {
  return value.toLocaleLowerCase("ja-JP").replace(/[\s　。、！？!?「」『』]/g, "");
}

function matchRecords(query) {
  const normalizedQuery = normalize(query);
  return records
    .map((record) => {
      const terms = searchTerms[record.memory_id] || [];
      const matchedTerms = terms.filter((term) => normalizedQuery.includes(normalize(term)));
      return { record, matchedTerms, score: matchedTerms.reduce((total, term) => total + term.length, 0) };
    })
    .filter((item) => item.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, 3);
}

/**
 * Confirmed memory records that `text` is clearly about, without the intent
 * check. Feed gates use this to allow first-person claims only when a reviewed
 * memory backs them. A record counts only when a matched term of two or more
 * characters also appears in the record itself (民謡, TextBlur, ナナチキ), so
 * generic search verbs such as 使って or 作った never back a claim.
 * It does not change getMemoryContext().
 */
export function findMemoryRecords(text) {
  return matchRecords(String(text || ""))
    .map(({ record, matchedTerms }) => {
      const content = normalize(record.content);
      const anchors = matchedTerms.filter((term) => [...term].length >= 2 && content.includes(normalize(term)));
      return { memory_id: record.memory_id, content: record.content, terms: matchedTerms, anchors };
    })
    .filter((item) => item.anchors.length > 0);
}

export function getMemoryContext(query) {
  if (!MEMORY_INTENT.test(query)) return "";

  const matches = matchRecords(query);

  if (matches.length === 0) {
    return [
      "今回の記憶検索では、質問に該当する本人の確認済み記録が見つからなかった。",
      "本人の好みや過去について推測せず、記録がないことを伝える。",
    ].join("\n");
  }

  const facts = matches.map(({ record }) => {
    const freshness = record.could_be_stale ? "（当時の情報で、現在は変わっている可能性がある）" : "";
    return `- ${record.content}${freshness} [source_message_id: ${record.source_message_id}; confidence: ${record.confidence}]`;
  });
  return `今回の質問に関係する本人の確認済み記憶候補:\n${facts.join("\n")}\n記憶候補と質問の話題が一致するときだけ使う。`;
}
