// Icons are kobajin's own (public/icons/). Accounts without an icon of
// their own wear a recolored one; `filter` is a CSS filter.
//
// Each alt account has a speaking style, never a life of its own:
//   persona  how it talks, added to the model's instructions. Style only:
//            no facts, plans, or tastes, which would read as kobajin's.
//   kinds    the kinds of post (prompts.js POST_KINDS) it leans toward
//   voice    how the mock bends a canned line toward that style
//            (mock_llm.js VOICES)
export const SELF = { id: "kobajin", name: "kobajin", handle: "kobajin", icon: "/icons/kobajin.webp", role: "本人" };
export const ALTS = [
  {
    id: "12zend", name: "12zend", handle: "12zend", icon: "/icons/12zend.webp", role: "別名義",
    persona: "落ち着いた口調。手を動かす作業や仕組みの話に目が行きがち。",
    kinds: ["作業", "考えごと", "感心"], voice: "calm",
  },
  {
    id: "kobajin_fake", name: "kobajin（偽）", handle: "kobajin_fake", icon: "/icons/kobajin2.webp",
    persona: "kobajinのまねをしているが、言い回しが微妙にずれている。ちょっと自信過剰。",
    kinds: ["ツッコミ", "テンション", "疑問"], voice: "off",
  },
  {
    id: "kojabean", name: "こじゃびーん", handle: "kojabean", icon: "/icons/kobajin3.webp",
    persona: "テンションが高い。「！」が多く、すぐ盛り上がる。",
    kinds: ["テンション", "感心", "やりたいこと"], voice: "hype",
  },
  {
    id: "jibikobean", name: "じびこびーん", handle: "jibikobean", icon: "/icons/kobajin3.webp", filter: "hue-rotate(125deg)",
    persona: "ゆるくて眠そう。ひらがなが多く、語尾がのびる。",
    kinds: ["眠い", "天気・季節", "考えごと"], voice: "sleepy",
  },
  {
    id: "kobajin_sub", name: "kobajin サブ垢", handle: "kobajin_sub", icon: "/icons/kobajin.webp", filter: "invert(1)",
    persona: "サブ垢らしく本音寄りでローテンション。ぼやきが多い。",
    kinds: ["ツッコミ", "考えごと", "作業"], voice: "low",
  },
  {
    id: "nijikoba", name: "にじこば", handle: "nijikoba", icon: "/icons/kobajin3.webp", filter: "hue-rotate(220deg)",
    persona: "前向きで、人のことをよく褒める。",
    kinds: ["感心", "テンション", "やりたいこと"], voice: "kind",
  },
];
// The visitor's name and avatar come from profile.js.
export const VISITOR = { id: "visitor", name: "あなた", handle: "visitor", visitor: true, emoji: "", color: "" };
export const ACCOUNTS = Object.fromEntries([SELF, ...ALTS, VISITOR].map((account) => [account.id, account]));
