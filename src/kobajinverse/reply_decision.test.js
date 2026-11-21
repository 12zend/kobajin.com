import assert from "node:assert/strict";
import { test } from "node:test";
import { replyNeed } from "./reply_decision.js";

const kindOf = (text) => replyNeed(text).kind;

test("lines that close an exchange need no answer", () => {
  for (const [text, kind] of [
    ["わかる", "agree"],
    ["わかる〜！", "agree"],
    ["わかるわ", "agree"],
    ["それなw", "agree"],
    ["それはそう", "agree"],
    ["たしかに", "agree"],
    ["なるほどね", "agree"],
    ["あーね", "agree"],
    ["うんうん", "agree"],
    ["@kobajin_sub わかる", "agree"],
    ["ありがとう！", "thanks"],
    ["ありがとうございます", "thanks"],
    ["あざす", "thanks"],
    ["草", "laugh"],
    ["wwww", "laugh"],
    ["それは草", "laugh"],
    ["天才", "praise"],
    ["えらい！", "praise"],
    ["おつかれさまです", "cheer"],
    ["がんばれ〜", "cheer"],
    ["👍", "reaction"],
    ["😂😂", "reaction"],
    ["！！", "reaction"],
  ]) {
    const result = replyNeed(text);
    assert.equal(result.kind, kind, text);
    assert.equal(result.need, "optional", text);
    assert.ok(result.chance < 1, text);
  }
});

test("questions, requests, greetings, and new content get an answer", () => {
  for (const [text, kind] of [
    ["具体的に", "detail"],
    ["詳しく", "detail"],
    ["どういうこと？", "detail"],
    ["なんで？", "question"],
    ["わかる？", "question"],
    ["それな？", "question"],
    ["何作るん", "question"],
    ["sb3ファイルとか？", "question"],
    ["おはよう", "greeting"],
    ["おやすみ", "greeting"],
    ["教えて", "request"],
    ["一緒にやろうよ、教えて", "request"],
    ["わかる、それ情報量多いよな", "statement"],
    ["今日ラーメン食べた", "statement"],
    ["情報量が多い…", "statement"],
    ["死ね", "statement"],
  ]) {
    const result = replyNeed(text);
    assert.equal(result.kind, kind, text);
    assert.equal(result.need, "reply", text);
    assert.equal(result.chance, 1, text);
  }
});

test("a reaction-only line offers its own emoji", () => {
  assert.deepEqual(replyNeed("🔥🔥").emoji, ["🔥"]);
  assert.deepEqual(replyNeed("わかる").emoji[0], "👍");
});

test("an empty line needs nothing", () => {
  assert.equal(kindOf(""), "empty");
  assert.equal(kindOf("@kobajin"), "empty");
});
