// Whether a line needs an answer.
//
// Not every reply asks for one. 「わかる」 under a post, 「ありがと」, 「草」,
// or a lone emoji closes the exchange: on X or Discord the author would
// usually leave a reaction and move on. A question, a request, a greeting,
// 「具体的に」, or a line that adds something new asks for an answer.
//
//     replyNeed(text) -> { need, kind, chance, emoji }
//
//   need    "reply": answer it. "optional": an answer is not needed.
//   kind    why (see KINDS below), for logs and for callers that weigh
//           kinds differently (banter in actions.js).
//   chance  how often to answer anyway, 0..1; 1 for "reply". A short
//           answer to 「ありがと」 now and then still reads naturally.
//   emoji   reactions that fit when there is no answer, best first.
//
// Rules only, no model call: it runs before anything is generated, the
// same for the mock and every kobajinAI version, and costs no GPU time.
// Anything it does not recognise is a "statement" and gets an answer, as
// before this existed. It has no imports, so `npm test` runs it in Node.

const MENTION = /[@＠][A-Za-z0-9_]+/g;
const EMOJI = /\p{Extended_Pictographic}/gu;
const EMOJI_PARTS = /[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}️‍]/gu;

// Asked about the account's own line: say more (brain.js `elaborate`).
export const DETAIL = /具体|詳し|くわし|kwsk|詳細|例えば|たとえば|例は|例を|どういう(こと|意味)|どゆこと|どういうことや|つまり|説明して|掘り下げ|もっと(教えて|聞かせて|言って)|もう(少し|ちょい|ちょっと)(詳|くわ|教|説明)/i;
// A question, by mark or question word. いつも and なんやろ are not questions.
export const QUESTION = /[?？]|なんで|なぜ|どう|どこ|いつ(?!も)|だれ|誰|なに|何|なんな|なんです|なんや(?!ろ)|どっち|どれ|どんな/;
// A guess that wants a yes or no (sb3ファイルとか？, 〜ってこと).
export const GUESS = /とか[?？]|って(こと)?[?？]|(ってこと|のこと|ですか|すか)[?？]?$|みたいな[?？]/;
const GREETING = /おはよ|おやすみ|こんにちは|こんちゃ|こんばんは|ただいま|おかえり|はじめまして|初めまして|よろしく|ハロー|hello|やっほ|ヤッホー|おっす|よっす|^やあ|またね|じゃあね|ばいばい|バイバイ/i;
const REQUEST = /(して|しよ|しよう|してや|してよ|してな|してほしい|してください|教えて|聞かせて|見せて|頼む|たのむ|お願い|おねがい|ちょうだい|くれ|ください)$/;

// Lines that close an exchange. Each is matched against the whole line,
// after mentions, emoji, spaces, punctuation, long vowels, and trailing
// laughter are taken off, so わかる〜！ and それなw count and
// わかる、それ情報量多いよな does not.
const ENDING = "(わ|な|ね|よ|よな|よね|やん|っす|です|ですね|ます|すぎる|すぎ)?";
const lexicon = (words) => new RegExp(`^(${words})${ENDING}$`, "i");
const KINDS = [
  ["thanks", lexicon("ありがと(う|う?ございます)?|ありがたい|あり|あざ(す|っす|ます)?|さんきゅ|サンキュ|thx|thanks|ty|感謝"), 0.3, ["❤️", "👍"]],
  ["agree", lexicon("わかる|わかりみ|わかります|それな|それはそう|ほんまそれ|ほんとそれ|まじそれ|マジそれ|たしかに|確かに|せやな|せやね|せやろ|せやんな|そうやな|そうだね|そうだよね|そうよな|そう|うん|うんうん|ね|なるほど|なる|了解|りょ|りょうかい|おけ|おっけ|ok|おk|いいね|いいよ|やろ|でしょ|だよね|よな|それある|あるある|同じく|同感|ほんま|まじか|マジか|へ|へえ|ほう|ふむ|あーね|あね"), 0.3, ["👍"]],
  ["praise", lexicon("天才|神|すごい|すご|えらい|最高|強い|つよ|さすが|かわいい|きれい|かっこいい"), 0.4, ["❤️", "👍"]],
  ["cheer", lexicon("おつかれ(さま)?|お疲れ(様)?|おつ|がんばれ|頑張れ|がんば|ファイト|応援してる"), 0.4, ["❤️", "👍"]],
  ["laugh", lexicon("それは草|わろた|ワロタ|わら|おもろ|おもしろ|面白い|じわる|ウケる|うける"), 0.25, ["👍", "❤️"]],
];
const LAUGH = /^(草|笑|w|ｗ)+$/i;
const TRAILING_LAUGH = /(草|笑|w|ｗ)+$/i;

function core(text) {
  return text.replace(MENTION, "").replace(EMOJI_PARTS, "").replace(/[\s。、．，.,!！…〜~ー―・「」『』()（）]+/g, "");
}

const need = (kind, chance = 1, emoji = []) => ({ need: chance >= 1 ? "reply" : "optional", kind, chance, emoji });

export function replyNeed(text) {
  const raw = String(text || "").replace(MENTION, "").trim();
  if (!raw) return need("empty", 0, ["👍"]);
  if (DETAIL.test(raw)) return need("detail");
  if (QUESTION.test(raw) || GUESS.test(raw)) return need("question");
  if (GREETING.test(raw)) return need("greeting");

  const bare = core(raw);
  // Only emoji or marks (👍, ！！, …): answer in kind, now and then a word.
  if (!bare) return need("reaction", 0.1, [...new Set(raw.match(EMOJI) || [])]);
  if (LAUGH.test(bare)) return need("laugh", 0.25, ["👍", "❤️"]);
  if (REQUEST.test(bare)) return need("request");
  const plain = bare.replace(TRAILING_LAUGH, "") || bare;
  for (const [kind, pattern, chance, emoji] of KINDS) {
    if (pattern.test(bare) || pattern.test(plain)) return need(kind, chance, emoji);
  }
  return need("statement");
}
