import { getMemoryContext } from "../memory.js";
// The same file chat_request.js reads; imported directly so the page does
// not pull in WebLLM until a visitor starts the model.
import SYSTEM_PROMPT_SOURCE from "../../self_twin/inference/system_prompt.txt?raw";

const SYSTEM_PROMPT = SYSTEM_PROMPT_SOURCE.trim();

// What each task asks of kobajinAI. llm.js sends PROMPTS[task].messages(input)
// with PROMPTS[task].decoding; the mock ignores both.
//
// The kobajinAI versions were trained on one shape: the site's system
// prompt (self_twin/inference/system_prompt.txt) and a group-chat log,
// "MM-DD HH:MM 【参加者A】…", answered as 自分. The tasks where kobajin
// speaks (post, reply, elaborate) keep that shape and add a short note on
// this place and the account's style. The tasks that only classify (react,
// judge, trend) use a plain instruction instead.

// The only emoji a reaction may use (the owner chose these six on
// 2026-11-21). The react prompt and the picker list them, brain.js falls
// back to 🤔 for anything else, and saved reactions with other emoji are
// dropped on load (store.js).
export const REACTION_EMOJI = ["🤔", "❓", "👍", "❤️", "⁉️", "😥"];

// The reaction emoji a text contains, or "". ❤ and ⁉ come with or without
// the emoji variation selector, so it is ignored when comparing.
const bare = (text) => String(text).replace(/\uFE0F/g, "");
export function findReaction(text) {
  const plain = bare(text);
  return REACTION_EMOJI.find((emoji) => plain.includes(bare(emoji))) || "";
}

// Kinds of post. Each auto post asks for a kind the timeline has not
// used lately, so the feed does not settle on one mood.
export const POST_KINDS = ["疑問", "ツッコミ", "感心", "テンション", "作業", "考えごと", "天気・季節", "やりたいこと", "眠い"];

const KIND_CUES = {
  "疑問": "ふと疑問に思ったこと",
  "ツッコミ": "思わずツッコみたくなったこと",
  "感心": "感心したこと",
  "テンション": "いまのテンション",
  "作業": "作業中に思ったこと",
  "考えごと": "考えごと",
  "天気・季節": "天気や季節のこと",
  "やりたいこと": "やってみたいこと（予定は作らない）",
  "眠い": "眠さ",
};

const stamp = (date = new Date()) => {
  const two = (n) => String(n).padStart(2, "0");
  return `${two(date.getMonth() + 1)}-${two(date.getDate())} ${two(date.getHours())}:${two(date.getMinutes())}`;
};
const log = (lines) => lines.map(([speaker, text]) => `${stamp()} 【${speaker}】${text}`).join("\n");

const PLACE = "ここはSNS。返事は1行だけで、ハッシュタグ・URL・絵文字は付けない。記憶欄にない自分の経験・予定・好みは作らず、決まっていないことは決まっていないと言う。暴言や物騒な投稿には同調も復唱もせず、軽くツッコむか心配する。";
const styleLine = (persona) => (persona ? `\nこのアカウントの口調: ${persona}` : "");
// The latest few are enough to avoid repeats; a long thread would
// otherwise make every prompt, and its prefill, longer and longer.
const othersLine = (others) => (others?.length ? `\nほかの人がもう返したこと（同じことは言わない）: ${others.slice(-8).join(" / ")}` : "");

function system(note, { persona, memoryQuery = "", others } = {}) {
  const memory = memoryQuery ? getMemoryContext(memoryQuery) : "";
  return [SYSTEM_PROMPT, memory, `${PLACE}\n${note}${styleLine(persona)}${othersLine(others)}`].filter(Boolean).join("\n\n");
}

export const PROMPTS = {
  post: {
    messages: ({ recent, kind, persona }) => [
      { role: "system", content: system(`今回はタイムラインへの独り言。10文字前後で1つだけ。${recent.length ? `\n最近の自分の投稿（同じ言い回しや話題は避ける）: ${recent.join(" / ")}` : ""}`, { persona }) },
      { role: "user", content: log([["参加者A", `${KIND_CUES[kind] || "いま思ったこと"}を一言でつぶやいて`]]) },
    ],
    decoding: { temperature: 0.9, top_p: 1, repetition_penalty: 1.12, max_tokens: 40 },
  },
  // `parent` is set when the text is itself a reply: what it answers.
  // `addressed` means the reply was aimed at this account, so `parent` is
  // its own line and a question needs an answer.
  reply: {
    messages: ({ text, parent, addressed, others, persona }) => [
      { role: "system", content: system("今回は投稿への返信。10文字前後で短く、相手の言葉を拾って反応する。自分に向けた質問には相づちでごまかさず、短く答える。", { persona, others, memoryQuery: addressed ? text : "" }) },
      {
        role: "user",
        content: parent
          ? addressed
            ? log([["自分", parent], ["参加者A", text]])
            : log([["参加者A", parent], ["参加者B", text]])
          : log([["参加者A", text]]),
      },
    ],
    decoding: { temperature: 0.8, top_p: 1, repetition_penalty: 1.12, max_tokens: 32 },
  },
  // Asked 「具体的に」「詳しく」 about its own line: say what it meant,
  // longer than a reply, without making up kobajin's life.
  elaborate: {
    messages: ({ text, parent, others, persona }) => [
      { role: "system", content: system("自分の発言に「具体的に」「詳しく」と聞かれた。何のことか・具体例を1つ、25文字以内で答える。相づちや聞き返しでごまかさない。", { persona, others, memoryQuery: text }) },
      { role: "user", content: log([["自分", parent], ["参加者A", text]]) },
    ],
    decoding: { temperature: 0.8, top_p: 1, repetition_penalty: 1.12, max_tokens: 64 },
  },
  react: {
    messages: ({ text }) => [
      { role: "system", content: `次の投稿に絵文字1つだけでリアクションする。候補: ${REACTION_EMOJI.join(" ")}。投稿の内容と温度に合うものを選び、絵文字だけを出す。` },
      { role: "user", content: `投稿: ${text}` },
    ],
    // Room for a whole emoji: 6 tokens could cut one in half (�).
    decoding: { temperature: 0.7, top_p: 1, repetition_penalty: 1.0, max_tokens: 12 },
  },
  judge: {
    messages: ({ text }) => [
      { role: "system", content: "次の投稿に共感できるか、面白いと思ったら「はい」、そうでなければ「いいえ」とだけ答える。暴言や物騒な投稿は「いいえ」。" },
      { role: "user", content: `投稿: ${text}` },
    ],
    decoding: { temperature: 0, top_p: 1, repetition_penalty: 1.0, max_tokens: 4 },
  },
  trend: {
    messages: ({ count }) => [
      { role: "system", content: "くだけた日本語の口癖や、日常で気になる言葉をハッシュタグにする。#から始めて1行に1つ、各12文字以内。似た意味のタグを並べない。人名・地名・商品名は使わない。" },
      { role: "user", content: `${count}個書いて。` },
    ],
    decoding: { temperature: 0.9, top_p: 1, repetition_penalty: 1.12, max_tokens: 120 },
  },
};
