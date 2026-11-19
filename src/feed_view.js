// Feed view: renders kobajinAI's local timeline, runs the idle-time autoposter,
// and answers replies (mode=reply). Chat (mode=chat) stays in main.js.

import { composeReply, composeTweet, RoleAbortedError } from "./agents.js";
import { createPost, openFeedStore, summarizeLogs } from "./feed.js";
import { fetchArticle, WIKIPEDIA_LICENSE } from "./wiki.js";

export const AUTO_POST_LIMIT = 3; // autonomous posts per browser session
const IDLE_MS = 45_000; // no input for this long counts as idle
const MIN_GAP_MS = 3 * 60_000; // between autonomous runs
const TICK_MS = 10_000;
const RECENT_ATTEMPT_WINDOW = 30; // dropped articles are not retried for this many runs
const SESSION_COUNT_KEY = "kobajinAI.feed.autoCount";

function readSessionCount() {
  try { return Number(sessionStorage.getItem(SESSION_COUNT_KEY)) || 0; } catch { return 0; }
}

function writeSessionCount(value) {
  try { sessionStorage.setItem(SESSION_COUNT_KEY, String(value)); } catch { /* in-memory count still applies */ }
}

function formatTime(timestamp) {
  const date = new Date(timestamp);
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay
    ? date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * options:
 *   scheduler   engine scheduler shared with chat
 *   getEngine() current WebLLM engine or null
 *   isChatBusy() true while chat is generating
 *   modelId     versioned model id for logs
 *   announce(text) screen-reader status
 */
export function initFeed({ scheduler, getEngine, isChatBusy, modelId, announce }) {
  const list = document.querySelector("#feed-list");
  const statusLine = document.querySelector("#feed-status");
  const postNowButton = document.querySelector("#feed-post-now");

  let store = null;
  let posts = [];
  let logs = [];
  let visible = false;
  let tweetJob = false;
  let jobLabel = "";
  let autoCount = readSessionCount();
  let lastActivity = Date.now();
  let lastAutoRun = 0;
  const openThreads = new Set();
  const replyingPosts = new Set();

  const storeReady = openFeedStore().then(async (opened) => {
    store = opened;
    posts = await store.listPosts();
    logs = await store.listLogs();
    render();
    return store;
  });

  for (const type of ["pointerdown", "keydown", "input", "wheel", "touchstart"]) {
    document.addEventListener(type, () => { lastActivity = Date.now(); }, { passive: true, capture: true });
  }

  function updateStatus() {
    const engine = getEngine();
    const summary = summarizeLogs(logs);
    const parts = [];
    if (tweetJob) parts.push(jobLabel || "Writing a post…");
    else if (!engine) parts.push("Load the model to let kobajinAI post");
    else if (autoCount >= AUTO_POST_LIMIT) parts.push("Auto-posting paused for this session");
    else parts.push("Posts on its own while you're idle");
    parts.push(`Auto ${autoCount}/${AUTO_POST_LIMIT}`);
    if (summary.runs) parts.push(`Gates passed ${summary.accepted}/${summary.runs}`);
    statusLine.textContent = parts.join(" · ");
    postNowButton.disabled = !engine || tweetJob;
  }

  function renderThread(post, container) {
    container.replaceChildren();
    for (const reply of post.replies) {
      const row = element("div", `thread-message ${reply.role}`);
      row.append(element("div", "thread-text", reply.content));
      container.append(row);
    }
    if (replyingPosts.has(post.id)) {
      const row = element("div", "thread-message assistant");
      row.append(element("div", "thread-text streaming"));
      container.append(row);
    }
    const form = element("form", "thread-form");
    form.dataset.postId = post.id;
    const input = element("textarea");
    input.rows = 1;
    input.placeholder = "Reply to kobajinAI…";
    input.setAttribute("aria-label", "Reply to this post");
    input.disabled = replyingPosts.has(post.id);
    const send = element("button", "thread-send", "Reply");
    send.type = "submit";
    send.disabled = replyingPosts.has(post.id) || !getEngine();
    form.append(input, send);
    container.append(form);
  }

  function render() {
    updateStatus();
    if (!list) return;
    list.replaceChildren();
    if (posts.length === 0) {
      list.append(element("p", "feed-empty", tweetJob
        ? "kobajinAI is reading something on Wikipedia…"
        : "No posts yet. kobajinAI posts here after reading a Wikipedia article."));
      return;
    }
    for (const post of posts) {
      const card = element("article", "feed-post");
      card.dataset.postId = post.id;

      const head = element("div", "feed-post-head");
      const avatar = element("img", "feed-avatar");
      avatar.src = "/logo.svg";
      avatar.alt = "";
      head.append(avatar, element("span", "feed-name", "kobajinAI"), element("span", "feed-time", formatTime(post.createdAt)));

      const text = element("p", "feed-text", post.text);

      const source = element("p", "feed-source");
      source.append("Source: ");
      const link = element("a", "", `Wikipedia「${post.source.title}」`);
      link.href = post.source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      source.append(link, ` · ${post.source.license || WIKIPEDIA_LICENSE}`);

      const actions = element("div", "feed-actions");
      const replyButton = element("button", "feed-reply-toggle", post.replies.length ? `Reply · ${post.replies.length}` : "Reply");
      replyButton.type = "button";
      replyButton.dataset.toggleThread = post.id;
      replyButton.setAttribute("aria-expanded", String(openThreads.has(post.id)));
      actions.append(replyButton);

      card.append(head, text, source, actions);
      if (openThreads.has(post.id)) {
        const thread = element("div", "feed-thread");
        renderThread(post, thread);
        card.append(thread);
      }
      list.append(card);
    }
  }

  async function runTweetJob(mode) {
    const engine = getEngine();
    if (!engine || tweetJob) return;
    tweetJob = true;
    jobLabel = "Reading Wikipedia…";
    render();
    const lease = await scheduler.acquire("tweet", { interruptible: true });
    try {
      if (mode === "auto" && (isChatBusy() || scheduler.priorityWaiting)) return;
      await storeReady;
      // Skip articles already posted and ones recently tried and dropped.
      const exclude = new Set([
        ...posts.map((post) => post.source.url),
        ...logs.slice(0, RECENT_ATTEMPT_WINDOW).map((log) => log.url).filter(Boolean),
      ]);
      const article = await fetchArticle({ exclude, signal: lease.signal });
      jobLabel = `Reading「${article.title}」…`;
      updateStatus();
      const result = await composeTweet(engine, article, {
        recentTexts: posts.map((post) => post.text),
        modelId,
        signal: lease.signal,
      });
      const logRow = await store.addLog({ ...result.log, accepted: result.accepted, mode });
      logs.unshift(logRow);
      if (mode === "auto") {
        autoCount += 1;
        writeSessionCount(autoCount);
      }
      if (result.accepted) {
        const post = createPost({ text: result.text, article, log: result.log, mode });
        await store.addPost(post);
        posts.unshift(post);
        announce?.("kobajinAI posted to the feed.");
      } else {
        announce?.("kobajinAI dropped a draft that did not pass its checks.");
      }
    } catch (error) {
      if (error instanceof RoleAbortedError || error?.name === "AbortError") {
        if (store) logs.unshift(await store.addLog({ interrupted: true, mode, model_id: modelId }));
      } else {
        console.error("kobajinAI feed post failed", error);
        announce?.("kobajinAI could not write a post this time.");
      }
    } finally {
      lease.release();
      tweetJob = false;
      jobLabel = "";
      if (mode === "auto") lastAutoRun = Date.now();
      render();
    }
  }

  async function sendReply(postId, text) {
    const post = posts.find((item) => item.id === postId);
    const engine = getEngine();
    const value = text.trim();
    if (!post || !engine || !value || replyingPosts.has(postId)) return;
    post.replies = [...post.replies, { role: "user", content: value, timestamp: Date.now() }];
    replyingPosts.add(postId);
    await store.updateReplies(postId, post.replies);
    render();
    const lease = await scheduler.acquire("reply");
    try {
      const result = await composeReply(engine, post, post.replies, { signal: lease.signal });
      post.replies = [...post.replies, { role: "assistant", content: result.text, timestamp: Date.now() }];
      await store.updateReplies(postId, post.replies);
      announce?.("kobajinAI replied.");
    } catch (error) {
      console.error("kobajinAI reply failed", error);
      announce?.("kobajinAI could not reply.");
    } finally {
      lease.release();
      replyingPosts.delete(postId);
      render();
      list.querySelector(`.thread-form[data-post-id="${CSS.escape(postId)}"] textarea`)?.focus();
    }
  }

  list.addEventListener("click", (event) => {
    const toggle = event.target.closest("[data-toggle-thread]");
    if (!toggle) return;
    const id = toggle.dataset.toggleThread;
    if (openThreads.has(id)) openThreads.delete(id);
    else openThreads.add(id);
    render();
    if (openThreads.has(id)) list.querySelector(`.thread-form[data-post-id="${CSS.escape(id)}"] textarea`)?.focus();
  });
  list.addEventListener("submit", (event) => {
    const form = event.target.closest(".thread-form");
    if (!form) return;
    event.preventDefault();
    void sendReply(form.dataset.postId, form.querySelector("textarea").value);
  });
  list.addEventListener("keydown", (event) => {
    if (event.target.matches?.(".thread-form textarea") && event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      event.target.form.requestSubmit();
    }
  });
  postNowButton.addEventListener("click", () => { void runTweetJob("manual"); });

  function autoTick() {
    const idle = Date.now() - lastActivity >= IDLE_MS;
    const prompt = document.querySelector("#prompt");
    const typing = Boolean(prompt?.value.trim()) || Boolean(list.querySelector(".thread-form textarea")?.value.trim());
    if (
      document.visibilityState === "visible"
      && getEngine()
      && idle
      && !typing
      && !tweetJob
      && !isChatBusy()
      && !scheduler.busy
      && replyingPosts.size === 0
      && autoCount < AUTO_POST_LIMIT
      && Date.now() - lastAutoRun >= MIN_GAP_MS
    ) {
      void runTweetJob("auto");
    } else {
      updateStatus();
    }
  }
  setInterval(autoTick, TICK_MS);

  return {
    setVisible(value) {
      visible = value;
      if (visible) render();
    },
    refresh: updateStatus,
  };
}
