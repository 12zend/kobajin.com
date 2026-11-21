// Icons are kobajin's own (public/icons/). Accounts without an icon of
// their own wear a recolored one; `filter` is a CSS filter.
export const SELF = { id: "kobajin", name: "kobajin", handle: "kobajin", icon: "/icons/kobajin.webp", role: "本人" };
export const ALTS = [
  { id: "12zend", name: "12zend", handle: "12zend", icon: "/icons/12zend.webp", role: "別名義" },
  { id: "kobajin_fake", name: "kobajin（偽）", handle: "kobajin_fake", icon: "/icons/kobajin2.webp" },
  { id: "kojabean", name: "こじゃびーん", handle: "kojabean", icon: "/icons/kobajin3.webp" },
  { id: "jibikobean", name: "じびこびーん", handle: "jibikobean", icon: "/icons/kobajin3.webp", filter: "hue-rotate(125deg)" },
  { id: "kobajin_sub", name: "kobajin サブ垢", handle: "kobajin_sub", icon: "/icons/kobajin.webp", filter: "invert(1)" },
  { id: "nijikoba", name: "にじこば", handle: "nijikoba", icon: "/icons/kobajin3.webp", filter: "hue-rotate(220deg)" },
];
export const VISITOR = { id: "visitor", name: "あなた", handle: "visitor", visitor: true };
export const ACCOUNTS = Object.fromEntries([SELF, ...ALTS, VISITOR].map((account) => [account.id, account]));
