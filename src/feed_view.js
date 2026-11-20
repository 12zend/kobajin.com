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
  const openChecks = new Set();
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
      // kobajin's line breaks read as separate messages, like the chat view.
      const lines = reply.role === "assistant"
        ? reply.content.split(/\n+/).map((line) => line.trim()).filter(Boolean)
        : [reply.content];
      for (const line of lines.length ? lines : [reply.content]) row.append(element("div", "thread-text", line));
      container.append(row);
    }
    if (replyingPosts.has(post.id)) {
      const row = element("div", "thread-message assistant");
      const typing = element("div", "thread-text streaming");
      typing.setAttribute("aria-label", "kobajinAI is replying");
      row.append(typing);
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

  const CHECK_LABELS = [
    ["form", "Form"],
    ["fact_rule", "Facts in source"],
    ["identity", "No invented experience"],
    ["novelty", "Not a repeat"],
    ["fact_model", "Model fact-check"],
  ];

  function checkPassed(key, value) {
    if (key === "fact_model") return value === "supported";
    if (key === "identity") return value === "pass" || value === "backed_by_memory";
    return value === "pass";
  }

  // The accepted attempt's gate results, as recorded in the post log.
  function renderChecks(post) {
    const verdicts = Array.isArray(post.log?.verdicts) ? post.log.verdicts : [];
    const final = verdicts[verdicts.length - 1];
    if (!final) return null;
    const passed = CHECK_LABELS.filter(([key]) => checkPassed(key, final[key]));
    const details = element("details", "feed-checks");
    details.dataset.postId = post.id;
    details.open = openChecks.has(post.id);
    const summary = element("summary");
    const redrafts = Number(post.log?.retries) || 0;
    summary.append(
      element("span", "feed-checks-dot"),
      element("span", "", `${passed.length}/${CHECK_LABELS.length} checks passed${redrafts ? ` · after ${redrafts} redraft${redrafts === 1 ? "" : "s"}` : ""}`),
    );
    details.append(summary);
    const listNode = element("ul", "feed-check-list");
    for (const [key, label] of CHECK_LABELS) {
      const ok = checkPassed(key, final[key]);
      const item = element("li", ok ? "ok" : "miss", key === "identity" && final[key] === "backed_by_memory" ? "Backed by a reviewed memory" : label);
      listNode.append(item);
    }
    details.append(listNode);
    if (post.log?.seed?.angle) {
      const angle = element("p", "feed-angle");
      angle.append(element("span", "", "Angle the model picked: "), document.createTextNode(post.log.seed.angle));
      details.append(angle);
    }
    return details;
  }

  function renderWorkingCard() {
    const card = element("div", "feed-working");
    card.setAttribute("role", "status");
    const avatar = element("span", "feed-avatar");
    avatar.setAttribute("aria-hidden", "true");
    avatar.append(element("span", "mark"));
    const copy = element("div", "feed-working-copy");
    copy.append(element("p", "feed-working-label", jobLabel || "Writing a post…"), element("p", "feed-working-sub", "Drafts are checked before anything is posted. A draft that fails is dropped."));
    card.append(avatar, copy, element("span", "feed-working-scan"));
    return card;
  }

  function render() {
    updateStatus();
    if (!list) return;
    list.replaceChildren();
    if (tweetJob) list.append(renderWorkingCard());
    if (posts.length === 0) {
      if (!tweetJob) list.append(element("p", "feed-empty", "No posts yet. kobajinAI posts here after reading a Wikipedia article."));
      return;
    }
    for (const post of posts) {
      const card = element("article", "feed-post");
      card.dataset.postId = post.id;

      const head = element("div", "feed-post-head");
      const avatar = element("span", "feed-avatar");
      avatar.setAttribute("aria-hidden", "true");
      avatar.append(element("span", "mark"));
      head.append(avatar, element("span", "feed-name", "kobajinAI"), element("span", "feed-time", formatTime(post.createdAt)));

      const source = element("p", "feed-source");
      source.append(element("span", "feed-source-verb", "Read "));
      const link = element("a", "", `Wikipedia「${post.source.title}」`);
      link.href = post.source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      source.append(link, element("span", "feed-license", ` · ${post.source.license || WIKIPEDIA_LICENSE}`));

      const text = element("p", "feed-text", post.text);
      const checks = renderChecks(post);

      const actions = element("div", "feed-actions");
      const replyButton = element("button", "feed-reply-toggle", post.replies.length ? `Reply · ${post.replies.length}` : "Reply");
      replyButton.type = "button";
      replyButton.dataset.toggleThread = post.id;
      replyButton.setAttribute("aria-expanded", String(openThreads.has(post.id)));
      actions.append(replyButton);

      card.append(head, source, text);
      if (checks) card.append(checks);
      card.append(actions);
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
      jobLabel = `Reading「${article.title}」and drafting…`;
      updateStatus();
      const label = list?.querySelector(".feed-working-label");
      if (label) label.textContent = jobLabel;
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
  // <details> toggle does not bubble; keep check lists open across re-renders.
  list.addEventListener("toggle", (event) => {
    const details = event.target;
    if (!details.matches?.(".feed-checks")) return;
    if (details.open) openChecks.add(details.dataset.postId);
    else openChecks.delete(details.dataset.postId);
  }, true);
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
