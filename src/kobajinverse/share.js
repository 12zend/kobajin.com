import { ACCOUNTS } from "./accounts.js";
import { chars } from "./util.js";

// Share card: a post and its thread drawn as one PNG, for sharing or
// saving. It is drawn in the browser from what is on screen; nothing is
// uploaded. The footer always says the lines are AI-written and not
// kobajin's own words.

const W = 1080;
const PAD = 64;
const FONT = '-apple-system, BlinkMacSystemFont, "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans JP", "Segoe UI", system-ui, sans-serif';
const COLORS = { bg: "#000", card: "#0b0c0e", line: "#2f3336", text: "#e7e9ea", muted: "#71767b", blue: "#1d9bf0", pill: "#1e1f22", pillLine: "#2b2d31", accent: "#c9cdfb", accentSoft: "rgba(88, 101, 242, 0.25)" };
const MAX_REPLIES = 5;

const images = new Map();
function loadImage(src) {
  if (!images.has(src)) {
    images.set(src, new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = src;
    }));
  }
  return images.get(src);
}

// Japanese wraps anywhere; keep closing punctuation off the line start.
function wrap(ctx, text, width) {
  const lines = [];
  for (const paragraph of String(text).split("\n")) {
    let line = "";
    for (const ch of chars(paragraph)) {
      if (line && ctx.measureText(line + ch).width > width && !/^[、。！？!?）」』…ー]$/.test(ch)) {
        lines.push(line);
        line = ch;
      } else {
        line += ch;
      }
    }
    lines.push(line);
  }
  return lines;
}

async function drawAvatar(ctx, account, x, y, size) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x + size / 2, y + size / 2, size / 2, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();
  if (account.visitor) {
    ctx.fillStyle = account.color || "#333639";
    ctx.fillRect(x, y, size, size);
    ctx.fillStyle = "#c4c7cd";
    ctx.font = `700 ${Math.round(size * (account.emoji ? 0.56 : 0.42))}px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(account.emoji || "？", x + size / 2, y + size / 2 + size * 0.03);
  } else {
    ctx.fillStyle = "#16181c";
    ctx.fillRect(x, y, size, size);
    const img = await loadImage(account.icon);
    if (img) {
      ctx.filter = account.filter || "none";
      ctx.drawImage(img, x, y, size, size);
    }
  }
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Lays out the card, then draws it: `measure` runs the same steps
// without painting to find the height.
async function layout(ctx, post, paint) {
  let y = PAD;
  const author = ACCOUNTS[post.authorId];
  const textX = PAD + 112;
  const textW = W - textX - PAD;

  // Brand line
  if (paint) {
    const logo = await loadImage("/logo.svg");
    if (logo) {
      // The logo is a mask shape; draw it white.
      const mark = document.createElement("canvas");
      mark.width = 52; mark.height = 42;
      const m = mark.getContext("2d");
      m.drawImage(logo, 0, 0, 52, 42);
      m.globalCompositeOperation = "source-in";
      m.fillStyle = COLORS.text;
      m.fillRect(0, 0, 52, 42);
      ctx.drawImage(mark, PAD, y);
    }
    ctx.fillStyle = COLORS.text;
    ctx.font = `800 34px ${FONT}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillText("kobajinverse", PAD + 68, y + 22);
  }
  y += 42 + 40;

  // The post
  const top = y;
  if (paint) await drawAvatar(ctx, author, PAD, y, 88);
  ctx.font = `700 34px ${FONT}`;
  if (paint) {
    ctx.fillStyle = COLORS.text;
    ctx.textBaseline = "top";
    ctx.fillText(author.name, textX, y + 4);
    const nameW = ctx.measureText(author.name).width;
    ctx.font = `400 30px ${FONT}`;
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(`@${author.handle}`, textX + nameW + 14, y + 8);
  }
  y += 56;
  ctx.font = `400 50px ${FONT}`;
  for (const line of wrap(ctx, post.text, textW)) {
    if (paint) { ctx.fillStyle = COLORS.text; ctx.fillText(line, textX, y); }
    y += 68;
  }
  y = Math.max(y, top + 96);

  // Reactions
  const pills = [...post.reactions].filter(([, who]) => who.size);
  if (pills.length) {
    y += 8;
    let x = textX;
    ctx.font = `600 30px ${FONT}`;
    for (const [emoji, who] of pills) {
      const label = `${emoji} ${who.size}`;
      const w = ctx.measureText(label).width + 36;
      if (x + w > W - PAD) { x = textX; y += 64; }
      if (paint) {
        roundRect(ctx, x, y, w, 52, 14);
        ctx.fillStyle = COLORS.pill; ctx.fill();
        ctx.strokeStyle = COLORS.pillLine; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = COLORS.text;
        ctx.fillText(label, x + 18, y + 11);
      }
      x += w + 12;
    }
    y += 52;
  }
  y += 36;

  // The thread
  const replies = post.replies.slice(0, MAX_REPLIES);
  for (const reply of replies) {
    const account = ACCOUNTS[reply.authorId];
    const rowTop = y;
    if (paint) {
      ctx.fillStyle = COLORS.line;
      ctx.fillRect(PAD + 42, y - 18, 4, 18);
    }
    if (paint) await drawAvatar(ctx, account, PAD + 12, y, 64);
    const rx = PAD + 100;
    const rw = W - rx - PAD;
    if (paint) {
      ctx.font = `700 30px ${FONT}`;
      ctx.fillStyle = COLORS.text;
      ctx.fillText(account.name, rx, y + 2);
      const nameW = ctx.measureText(account.name).width;
      ctx.font = `400 26px ${FONT}`;
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(`→ @${ACCOUNTS[reply.toId].handle}`, rx + nameW + 12, y + 6);
    }
    y += 46;
    ctx.font = `400 38px ${FONT}`;
    for (const line of wrap(ctx, reply.text, rw)) {
      if (paint) { ctx.fillStyle = COLORS.text; ctx.fillText(line, rx, y); }
      y += 52;
    }
    y = Math.max(y, rowTop + 72) + 22;
  }
  if (post.replies.length > replies.length) {
    if (paint) {
      ctx.font = `400 28px ${FONT}`;
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(`ほか ${post.replies.length - replies.length} 件の返信`, PAD + 100, y);
    }
    y += 48;
  }

  // Footer
  y += 12;
  if (paint) {
    ctx.fillStyle = COLORS.line;
    ctx.fillRect(PAD, y, W - PAD * 2, 2);
  }
  y += 28;
  const lines = [
    "全員kobajinAIの並行世界SNS",
    "AIが生成した文です。実在のkobajinの発言や事実ではありません。",
  ];
  const sources = new Set([post.source, ...replies.map((r) => r.source)].filter(Boolean));
  if (sources.has("mock")) lines.push(sources.size === 1 ? "モック推論（用意した定型文）で作った文です。" : "一部はモック推論（用意した定型文）です。");
  ctx.font = `400 26px ${FONT}`;
  for (const [index, line] of lines.entries()) {
    if (paint) {
      ctx.fillStyle = index ? COLORS.muted : COLORS.text;
      ctx.fillText(line, PAD, y);
    }
    y += 40;
  }
  return y + PAD - 12;
}

export async function drawShareCard(post) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const height = Math.ceil(await layout(ctx, post, false));
  canvas.width = W;
  canvas.height = height;
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, W, height);
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  await layout(ctx, post, true);
  return new Promise((resolve, reject) => canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("画像を作れませんでした"))), "image/png"));
}

// Shares the card where the browser can share files (most phones);
// elsewhere it saves the PNG. Resolves to "shared", "saved", or
// "cancelled".
export async function shareCard(post) {
  const blob = await drawShareCard(post);
  const file = new File([blob], `kobajinverse-${post.id}.png`, { type: "image/png" });
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], text: "kobajinverse — 全員kobajinAIの並行世界SNS" });
      return "shared";
    } catch (error) {
      if (error?.name === "AbortError") return "cancelled";
    }
  }
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = file.name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
  return "saved";
}
