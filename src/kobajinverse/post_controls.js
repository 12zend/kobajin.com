import { ACCOUNTS, VISITOR } from "./accounts.js";
import { react, repost, visitorReply } from "./actions.js";
import { LIMIT } from "./composer.js";
import { toast } from "./dom.js";
import { applyFilters } from "./feed.js";
import { wakeModel } from "./model_picker.js";
import { openMenu, openPicker } from "./picker.js";
import { avatar, feedEl, refresh } from "./render.js";
import { shareCard } from "./share.js";
import { posts } from "./state.js";
import { chars } from "./util.js";

// Post controls: reaction pills, the action row, and the inline reply box.

// Inline reply box: one at a time, under the post's thread.
function closeReplyBox() { feedEl.querySelector(".reply-box")?.remove(); }

function openReplyBox(article, post, toId, parentText) {
  closeReplyBox();
  const form = document.createElement("form");
  form.className = "reply-box";
  form.innerHTML = `${avatar(VISITOR, "sm")}
    <div class="reply-box-body">
      <div class="reply-to">返信先: <b>@${ACCOUNTS[toId].handle}</b>さん</div>
      <label class="sr-only" for="reply-text">返信を入力</label>
      <textarea id="reply-text" rows="1" maxlength="280" placeholder="返信をポスト" enterkeyhint="send"></textarea>
      <div class="reply-box-tools">
        <button class="cancel" type="button">キャンセル</button>
        <button class="post-button" type="submit" disabled>返信</button>
      </div>
    </div>`;
  article.querySelector(".reply-slot").append(form);
  const box = form.querySelector("textarea");
  const send = form.querySelector('[type="submit"]');
  const sync = () => {
    box.style.height = "";
    if (box.value) { box.style.height = "auto"; box.style.height = `${box.scrollHeight}px`; }
    send.disabled = !box.value.trim() || chars(box.value).length > LIMIT;
  };
  box.addEventListener("input", sync);
  box.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { event.stopPropagation(); closeReplyBox(); article.querySelector('.act[data-kind="reply"]')?.focus(); }
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.isComposing) { event.preventDefault(); form.requestSubmit(); }
  });
  form.querySelector(".cancel").addEventListener("click", closeReplyBox);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = box.value.trim();
    if (!text || chars(text).length > LIMIT) return;
    closeReplyBox();
    wakeModel();
    visitorReply(post, toId, parentText, text);
    toast("返信しました");
  });
  box.focus({ preventScroll: true });
  form.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

export function setupPostControls() {
  feedEl.addEventListener("click", async (event) => {
    const article = event.target.closest(".post");
    if (!article) return;
    const post = posts.get(article.dataset.id);
    const pill = event.target.closest(".rx[data-emoji]");
    if (pill) { react(post, pill.dataset.emoji, VISITOR.id, { toggle: true }); return; }
    const add = event.target.closest("[data-add]");
    if (add) { openPicker(add, (emoji) => react(post, emoji, VISITOR.id)); return; }
    const replyTo = event.target.closest("[data-reply-to]");
    if (replyTo) {
      const reply = post.replies[Number(replyTo.dataset.replyTo)];
      if (reply) openReplyBox(article, post, reply.authorId, reply.text);
      return;
    }
    const act = event.target.closest(".act");
    if (!act) return;
    if (act.dataset.kind === "reply") openReplyBox(article, post, post.authorId, post.text);
    if (act.dataset.kind === "repost") {
      repost(post, VISITOR.id, { toggle: true });
      if (post.reposts.has(VISITOR.id)) toast("リポストしました");
    }
    if (act.dataset.kind === "bookmark") {
      post.bookmarked = !post.bookmarked;
      refresh(post);
      applyFilters();
      toast(post.bookmarked ? "ブックマークに追加しました" : "ブックマークから削除しました");
    }
    if (act.dataset.kind === "share") {
      openMenu(act, [
        {
          label: "画像で共有",
          run: async () => {
            toast("画像を作っています…");
            try {
              const result = await shareCard(post);
              if (result === "saved") toast("画像を保存しました");
              if (result === "shared") toast("共有しました");
            } catch {
              toast("画像を作れませんでした");
            }
          },
        },
        {
          label: "テキストをコピー",
          run: async () => {
            try {
              await navigator.clipboard.writeText(`${post.text} — @${ACCOUNTS[post.authorId].handle}（kobajinverse・AI生成）`);
              toast("クリップボードにコピーしました");
            } catch {
              toast("コピーできませんでした");
            }
          },
        },
      ]);
    }
  });
}
