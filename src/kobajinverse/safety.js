import { identityGate, noveltyGate } from "../gates.js";
import { findMemoryRecords } from "../memory.js";
import { chars } from "./util.js";

// Checks every generated line must pass before it is shown, for the mock
// and the real model alike. A line that fails is generated again (see
// brain.js) and dropped after the last try.

// Insults and threats. The lookaheads keep 少しねむい and カスタム out.
export const HARSH = /死ね|しね(?![ぁ-んァ-ヴ])|殺す|ころす|ぶっ殺|ぶっころ|ぶっとば|消えろ|きえろ|くたばれ|黙れ|だまれ|うざい|ウザい|きもい|キモい|カス(?![ァ-ヴ])|滅べ|滅びろ|爆破/;

// Leftovers of the training data's anonymized log format.
const LEAK = /【|】|\[source|source_message_id|参加者[A-ZＡ-Ｚ]|kobajinAI として|AIなので|AIです/;
// Links, hashtags, and @mentions. The training chats are full of
// "@通話募集" and friends' handles; a kobajin must not tag real people.
const LINK = /https?:\/\/|www\.|#|＃|@|＠/;

// Why a line is not shown, or "" when it may be shown.
//   recent: lines it must not repeat (the thread's other replies, or the
//           timeline's recent posts)
export function rejectReason(text, { recent = [], source = "" } = {}) {
  if (!text) return "empty";
  if (LEAK.test(text)) return "leak";
  if (HARSH.test(text)) return "harsh";
  if (LINK.test(text)) return "link";
  // kobajin's own past, habits, or tastes, unless a confirmed memory backs it.
  if (!identityGate(text, findMemoryRecords(text)).pass) return "identity";
  if (recent.includes(text)) return "repeat";
  // Short lines share bigrams easily, so only near-copies count.
  if (chars(text).length >= 6 && !noveltyGate(text, recent).pass) return "repeat";
  // Parroting the post back is not a reply.
  if (source && text.replace(/[\s。、！？!?]/g, "") === source.replace(/[\s。、！？!?]/g, "")) return "echo";
  return "";
}
