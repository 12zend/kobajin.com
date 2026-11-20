import {
  CreateWebWorkerMLCEngine,
  hasModelInCache,
  prebuiltAppConfig,
} from "@mlc-ai/web-llm";
import { getMemoryContext, lookupMemories } from "./memory.js";
import { createEngineScheduler } from "./agents.js";
import { initFeed } from "./feed_view.js";
import SYSTEM_PROMPT_SOURCE from "../self_twin/inference/system_prompt.txt?raw";

const BASE_MODEL_ID = "Qwen3.5-4B-q4f16_1-MLC";
const MODEL_REVISION = "a2534a20253d54cce925307fbd76086498b09a11";
const MODEL_ID = "kobajinAI-qwen35-4b-luna-v3-a2534a2-q4f16_1-MLC";
const MODEL_URL = import.meta.env.VITE_MODEL_URL || `https://huggingface.co/kobajin/kobajinAI/resolve/${MODEL_REVISION}/`;
const SYSTEM_PROMPT = SYSTEM_PROMPT_SOURCE.trim();

// Turn-taking: after the visitor sends, kobajinAI waits for a short pause so
// several quick messages are answered together, the way people chat.
const SETTLE_MS = 1200;
// A reload within this window reopens a chat that was still waiting for a reply.
const RESUME_WINDOW_MS = 30 * 60_000;
const TIME_DIVIDER_GAP_MS = 15 * 60_000;
const DEV_FLAGS = import.meta.env.DEV ? new URLSearchParams(location.search) : new URLSearchParams();
const USE_MOCK_ENGINE = DEV_FLAGS.has("mock-engine");
// `?no-webgpu` (dev only) previews the unsupported-browser state.
const HAS_WEBGPU = Boolean(navigator.gpu) && !DEV_FLAGS.has("no-webgpu");

const $ = (selector) => document.querySelector(selector);
const welcome = $("#welcome");
const messagesView = $("#messages");
const promptBox = $("#prompt");
const sendButton = $("#send");
const stopButton = $("#stop");
const loadButton = $("#load-model");
const loadLabel = $("#load-label");
const liveStatus = $("#live-status");
const conversationList = $("#conversation-list");
const historyCount = $("#history-count");
const chatTitle = $("#chat-title");
const presenceLine = $("#presence-line");
const arrival = $("#arrival");
const arrivalBar = $("#arrival-bar");
const progressDetail = $("#progress-detail");
const deviceTitle = $("#device-title");
const deviceDetail = $("#device-detail");
const composerHint = $("#composer-hint");
const sidebar = $("#sidebar");
const sidebarScrim = $("#sidebar-scrim");
const aboutDialog = $("#about-dialog");
const chatStage = $("#chat-stage");
const composerArea = $("#composer-area");
const feedStage = $("#feed-stage");
const feedToggle = $("#feed-toggle");
const chatsTab = $("#chats-tab");
const jumpLatest = $("#jump-latest");
const menuUnread = $("#menu-unread");

const conversationStorageKey = "kobajinAI.conversations.v1";
const engineScheduler = createEngineScheduler();
engineScheduler.setInterrupt(() => engine?.interruptGenerate());
const storagePersistence = requestStoragePersistence();

let engine = null;
let enginePromise = null;
let worker = null;
let modelCached = false;
let presence = "offline"; // offline | loading | online | unsupported | error
let feedVisible = false;

let threads = readSavedConversations();
let active = newThread();
const pending = new Set(); // thread ids waiting for a reply
const unread = new Set(); // thread ids with a reply the visitor has not seen
const failures = new Map(); // thread id -> error text for the last attempt
let settleTimer = null;
// The reply being written: { threadId, afterIndex, lines, stopped, startedAt }
let job = null;
let liveGroup = null;
// Only messages added since the last render of the same thread animate in.
let rendered = { threadId: null, count: 0 };

/* ---------- Storage ---------- */

function newThread() {
  return { id: globalThis.crypto?.randomUUID?.() || `chat-${Date.now()}`, title: "New chat", updatedAt: 0, messages: [] };
}

function sanitizeMemory(memory) {
  if (!memory || !["none", "empty", "found"].includes(memory.kind)) return undefined;
  const records = Array.isArray(memory.records) ? memory.records
    .filter((record) => record && typeof record.content === "string")
    .map(({ id, content, observedAt, stale }) => ({ id: String(id || ""), content, observedAt: observedAt || null, stale: Boolean(stale) })) : [];
  return { kind: memory.kind, records };
}

function sanitizeMessage(message) {
  const clean = { role: message.role, content: message.content, timestamp: Number(message.timestamp) || 0 };
  if (message.readAt) clean.readAt = Number(message.readAt) || undefined;
  if (message.stopped) clean.stopped = true;
  const memory = sanitizeMemory(message.memory);
  if (memory) clean.memory = memory;
  return clean;
}

function readSavedConversations() {
  try {
    const parsed = JSON.parse(localStorage.getItem(conversationStorageKey) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item) => item && typeof item.id === "string" && Array.isArray(item.messages))
      .map((item) => ({
        id: item.id,
        title: typeof item.title === "string" ? item.title : "New chat",
        updatedAt: Number(item.updatedAt) || 0,
        messages: item.messages
          .filter((message) => message && ["user", "assistant"].includes(message.role) && typeof message.content === "string")
          .map(sanitizeMessage),
      }))
      .filter((item) => item.messages.length > 0)
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, 40);
  } catch {
    return [];
  }
}

function persistThreads() {
  try {
    localStorage.setItem(conversationStorageKey, JSON.stringify(threads));
  } catch {
    liveStatus.textContent = "This chat could not be saved in browser storage.";
  }
}

// Threads are stored by reference, so `active` and its list entry are the same object.
function saveThread(thread) {
  if (!thread.messages.some((message) => message.role === "user")) return;
  const firstMessage = thread.messages.find((message) => message.role === "user")?.content || "New chat";
  thread.title = firstMessage.replace(/\s+/g, " ").trim().slice(0, 42) || "New chat";
  thread.updatedAt = Date.now();
  threads = [thread, ...threads.filter((item) => item.id !== thread.id)]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 40);
  persistThreads();
  renderConversationList();
}

async function requestStoragePersistence() {
  try {
    return await navigator.storage?.persist?.() ?? false;
  } catch {
    return false;
  }
}

/* ---------- Small helpers ---------- */

const isJapanese = (text) => /[぀-ヿ㐀-鿿]/.test(text);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function clockTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function shortWhen(timestamp) {
  const date = new Date(timestamp);
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay ? clockTime(timestamp) : date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function dividerLabel(timestamp) {
  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date(Date.now() - 86_400_000);
  const day = date.toDateString() === today.toDateString()
    ? "Today"
    : date.toDateString() === yesterday.toDateString()
      ? "Yesterday"
      : date.toLocaleDateString([], { month: "short", day: "numeric" });
  return `${day} ${clockTime(timestamp)}`;
}

function replyLines(content) {
  const lines = String(content || "").split(/\n+/).map((line) => line.trim()).filter(Boolean);
  return lines.length ? lines : [String(content || "")];
}

function needsReply(thread) {
  return thread.messages.length > 0 && thread.messages[thread.messages.length - 1].role === "user";
}

function trailingUserMessages(thread) {
  const result = [];
  for (let index = thread.messages.length - 1; index >= 0 && thread.messages[index].role === "user"; index -= 1) {
    result.unshift(thread.messages[index]);
  }
  return result;
}

function findThread(id) {
  if (active.id === id) return active;
  return threads.find((item) => item.id === id) || null;
}

/* ---------- Presence (where the model is) ---------- */

const progressState = { phase: null, fetchStartedAt: 0, lastEta: null };

function formatEta(seconds) {
  if (!Number.isFinite(seconds)) return "";
  if (seconds < 50) return "under a minute left";
  const minutes = Math.round(seconds / 60);
  return minutes <= 1 ? "about a minute left" : `about ${minutes} min left`;
}

function setStep(step, state, note) {
  const item = arrival.querySelector(`[data-step="${step}"]`);
  if (!item) return;
  item.classList.toggle("active", state === "active");
  item.classList.toggle("done", state === "done");
  if (note !== undefined) item.querySelector(".step-note").textContent = note;
}

function setPresence(state, { line, title, detail, button } = {}) {
  presence = state;
  document.body.dataset.presence = state;
  if (line) presenceLine.textContent = line;
  if (title) deviceTitle.textContent = title;
  if (detail !== undefined) deviceDetail.textContent = detail;
  if (button !== undefined) {
    loadLabel.textContent = button || "";
    loadButton.hidden = !button;
  }
  loadButton.disabled = state === "loading";
  updateComposerHint();
  updatePresenceLine();
}

// While online, the header line follows the conversation: typing, or online.
function updatePresenceLine() {
  if (feedVisible) return;
  if (presence !== "online") return;
  if (job && job.threadId === active.id) presenceLine.textContent = "typing…";
  else if (job) presenceLine.textContent = "Online · replying in another chat";
  else presenceLine.textContent = "Online · running on this device";
}

function updateComposerHint() {
  const hints = {
    offline: modelCached
      ? "The model is saved in this browser. Sending opens it."
      : "Your first message starts a one-time download (about 2.2 GiB). Keep writing while it loads.",
    loading: "Keep writing — your messages are delivered as soon as it's online.",
    online: "Send a few short messages if you like — it waits for a pause, then answers them together.",
    unsupported: "This browser can't run the model. Open the page in a recent desktop Chrome or Edge.",
    error: "The model didn't load. Your messages are kept — try bringing it online again.",
  };
  composerHint.textContent = hints[presence] || "";
}

function applyProgress(report) {
  const text = String(report?.text || "");
  const fraction = Number.isFinite(report?.progress) ? Math.max(0, Math.min(1, report.progress)) : 0;
  const phase = /^Fetching/i.test(text) ? "fetch" : /shader|GPU/i.test(text) && !/^Finish/i.test(text) ? "warm" : /^Finish/i.test(text) ? "done" : "load";
  const percent = Math.round(fraction * 100);
  progressState.phase = phase;
  arrival.hidden = false;

  let lineText = "";
  if (phase === "fetch") {
    if (!progressState.fetchStartedAt && fraction > 0) progressState.fetchStartedAt = performance.now();
    const elapsed = (performance.now() - progressState.fetchStartedAt) / 1000;
    const eta = fraction >= 0.02 && elapsed > 2 ? (elapsed * (1 - fraction)) / fraction : null;
    const megabytes = /([\d.]+)\s*MB\s*fetched/i.exec(text)?.[1];
    const amount = megabytes ? `${(Number(megabytes) / 1024).toFixed(1)} of 2.2 GiB` : `${percent}%`;
    const etaText = formatEta(eta);
    setStep("fetch", "active", [amount, etaText].filter(Boolean).join(" · "));
    setStep("load", "pending", "");
    setStep("warm", "pending", "");
    lineText = `Downloading ${percent}%${etaText ? ` · ${etaText}` : ""}`;
    arrivalBar.style.width = `${fraction * 85}%`;
    progressDetail.textContent = "Downloaded once, then kept in this browser. Later visits skip this step.";
  } else if (phase === "load") {
    setStep("fetch", "done", modelCached ? "Already saved here" : "Saved in this browser");
    setStep("load", "active", `${percent}%`);
    setStep("warm", "pending", "");
    lineText = `Loading into memory ${percent}%`;
    arrivalBar.style.width = `${(modelCached ? 0 : 85) + fraction * (modelCached ? 70 : 10)}%`;
    progressDetail.textContent = "Reading the saved model into this device's memory.";
  } else if (phase === "warm") {
    setStep("fetch", "done");
    setStep("load", "done", "Done");
    setStep("warm", "active", `${percent}%`);
    lineText = "Warming up the GPU…";
    arrivalBar.style.width = `${(modelCached ? 70 : 95) + fraction * (modelCached ? 30 : 5)}%`;
    progressDetail.textContent = "Preparing the GPU programs it uses to write.";
  } else {
    setStep("fetch", "done");
    setStep("load", "done", "Done");
    setStep("warm", "done", "Done");
    arrivalBar.style.width = "100%";
  }
  if (lineText) setPresence("loading", { line: lineText, title: "Coming online…", detail: lineText, button: "" });
}

function buildAppConfig() {
  const baseRecord = prebuiltAppConfig.model_list.find((item) => item.model_id === BASE_MODEL_ID);
  if (!baseRecord) return null;
  return {
    ...prebuiltAppConfig,
    model_list: [...prebuiltAppConfig.model_list, { ...baseRecord, model_id: MODEL_ID, model: MODEL_URL }],
  };
}

function markUnsupported() {
  setPresence("unsupported", {
    line: "Can't run in this browser · needs WebGPU",
    title: "Can't run in this browser",
    detail: "kobajinAI needs WebGPU. Try a recent desktop version of Chrome or Edge.",
    button: "",
  });
  promptBox.disabled = true;
  promptBox.placeholder = "This browser can't run kobajinAI";
  sendButton.disabled = true;
}

async function ensureModel() {
  if (engine) return engine;
  if (enginePromise) return enginePromise;
  if (!HAS_WEBGPU && !USE_MOCK_ENGINE) {
    markUnsupported();
    liveStatus.textContent = "WebGPU is not available in this browser.";
    return null;
  }
  const appConfig = buildAppConfig();
  if (!appConfig) {
    const message = `WebLLM ${BASE_MODEL_ID} runtime was not found. Refresh the app after updating its package.`;
    setPresence("error", { line: "Offline · this app version lacks the model runtime", title: "Model runtime unavailable", detail: message, button: "Try again" });
    liveStatus.textContent = message;
    return null;
  }

  progressState.fetchStartedAt = 0;
  arrival.hidden = false;
  setStep("fetch", modelCached ? "done" : "active", modelCached ? "Already saved here" : "Starting…");
  setStep("load", "pending", "");
  setStep("warm", "pending", "");
  arrivalBar.style.width = "0%";
  progressDetail.textContent = modelCached ? "Opening the model saved in this browser." : "Connecting to the model repository on Hugging Face.";
  setPresence("loading", { line: "Coming online…", title: "Coming online…", detail: "Connecting…", button: "" });
  liveStatus.textContent = "Loading the kobajinAI model into this browser.";
  void storagePersistence;

  enginePromise = (async () => {
    if (import.meta.env.DEV && USE_MOCK_ENGINE) {
      const { createMockEngine } = await import("./dev_mock_engine.js");
      engine = await createMockEngine(applyProgress, { cached: modelCached });
    } else {
      worker = new Worker(new URL("./webllm.worker.js", import.meta.url), { type: "module" });
      engine = await CreateWebWorkerMLCEngine(worker, MODEL_ID, { appConfig, initProgressCallback: applyProgress });
    }
    modelCached = true;
    applyProgress({ progress: 1, text: "Finish" });
    setPresence("online", { line: "Online · running on this device", title: "Online on this device", detail: "Replies are written here, in this browser.", button: "" });
    setTimeout(() => { if (presence === "online") arrival.hidden = true; }, 1600);
    liveStatus.textContent = "kobajinAI is online in this browser.";
    feed.refresh();
    if (needsReply(active) && !failures.has(active.id)) pending.add(active.id);
    schedulePump(job ? 0 : 400);
    return engine;
  })().catch((error) => {
    console.error("kobajinAI model load failed", error);
    enginePromise = null;
    engine = null;
    worker?.terminate();
    worker = null;
    setPresence("error", {
      line: "Offline · couldn't load the model",
      title: "Couldn't load the model",
      detail: error?.message || "Check your connection and available browser storage, then try again.",
      button: "Try again",
    });
    progressDetail.textContent = error?.message || "Check your connection and available browser storage, then try again.";
    liveStatus.textContent = "The local model could not be loaded.";
    renderThread();
    return null;
  });
  return enginePromise;
}

/* ---------- Thread rendering ---------- */

function memoryReceipt(memory) {
  if (!memory || memory.kind === "none") return null;
  if (memory.kind === "empty") {
    const note = element("span", "memory-receipt empty", "Checked memories · nothing confirmed");
    note.title = "No reviewed memory matched, so it should not guess about kobajin's past or tastes.";
    return note;
  }
  const details = element("details", "memory-receipt");
  const count = memory.records.length;
  details.append(element("summary", "", `Used ${count} reviewed ${count === 1 ? "memory" : "memories"}`));
  const list = element("ul");
  for (const record of memory.records) {
    const item = element("li", "", record.content);
    if (record.stale) item.append(element("span", "memory-stale", " (may be out of date)"));
    list.append(item);
  }
  details.append(list);
  return details;
}

function groupShell(role) {
  const group = element("div", `group ${role}`);
  if (role === "assistant") {
    const avatar = element("span", "group-avatar");
    avatar.setAttribute("aria-hidden", "true");
    avatar.append(element("span", "logo-mark"));
    group.append(avatar);
  }
  const body = element("div", "group-body");
  group.append(body);
  return { group, body };
}

function userReceipt(message) {
  if (message.readAt) return `Read ${clockTime(message.readAt)}`;
  if (failures.has(active.id)) return "Sent · not answered";
  if (presence === "online") return "Delivered";
  if (presence === "unsupported") return "Sent · can't be answered in this browser";
  return "Sent · it'll read this when it's online";
}

function renderLiveGroup() {
  if (!job || job.threadId !== active.id) {
    liveGroup?.remove();
    liveGroup = null;
    return;
  }
  if (!liveGroup) {
    const { group, body } = groupShell("assistant");
    group.classList.add("live");
    liveGroup = { group, body, typing: element("div", "bubble typing") };
    liveGroup.typing.setAttribute("aria-label", "kobajinAI is typing");
    liveGroup.typing.append(element("i"), element("i"), element("i"));
    body.append(liveGroup.typing);
  }
  const { body, typing } = liveGroup;
  const shown = body.querySelectorAll(".bubble:not(.typing)").length;
  for (const line of job.lines.slice(shown)) body.insertBefore(element("div", "bubble", line), typing);
}

function renderThread() {
  const stick = isNearBottom();
  messagesView.replaceChildren();
  liveGroup = null;
  const { messages } = active;
  welcome.classList.toggle("compact", messages.length > 0);
  const lastUserIndex = messages.map((message) => message.role).lastIndexOf("user");
  const freshFrom = rendered.threadId === active.id ? rendered.count : Infinity;
  rendered = { threadId: active.id, count: messages.length };
  let index = 0;
  let previousTime = 0;
  while (index < messages.length) {
    const role = messages[index].role;
    const { group, body } = groupShell(role);
    const first = messages[index];
    if (first.timestamp && (!previousTime || first.timestamp - previousTime >= TIME_DIVIDER_GAP_MS)) {
      messagesView.append(element("div", "time-divider", dividerLabel(first.timestamp)));
    }
    let last = first;
    while (index < messages.length && messages[index].role === role) {
      last = messages[index];
      const bubbleClass = index >= freshFrom ? "bubble fresh" : "bubble";
      if (index >= freshFrom) group.classList.add("fresh");
      if (role === "assistant") for (const line of replyLines(last.content)) body.append(element("div", bubbleClass, line));
      else body.append(element("div", bubbleClass, last.content));
      if (role === "assistant" && (last.memory || last.stopped)) {
        const meta = element("div", "group-meta");
        if (last.stopped) meta.append(element("span", "stopped-note", "Stopped"));
        const receipt = memoryReceipt(last.memory);
        if (receipt) meta.append(receipt);
        if (meta.childNodes.length) body.append(meta);
      }
      if (job && job.threadId === active.id && job.afterIndex === index) {
        index += 1;
        break;
      }
      index += 1;
    }
    previousTime = last.timestamp || previousTime;
    if (role === "user" && index - 1 === lastUserIndex) {
      body.append(element("div", "group-meta receipt", userReceipt(last)));
    }
    messagesView.append(group);
    if (job && job.threadId === active.id && job.afterIndex === index - 1) {
      renderLiveGroup();
      messagesView.append(liveGroup.group);
    }
  }
  if (failures.has(active.id) && !job) {
    const row = element("div", "thread-notice");
    row.append(element("span", "", failures.get(active.id)));
    const retry = element("button", "notice-action", "Try again");
    retry.type = "button";
    retry.addEventListener("click", () => {
      failures.delete(active.id);
      pending.add(active.id);
      renderThread();
      if (!engine) void ensureModel();
      schedulePump(0);
    });
    row.append(retry);
    messagesView.append(row);
  }
  syncControls();
  if (stick) scrollToLatest();
}

function isNearBottom() {
  return chatStage.scrollHeight - chatStage.scrollTop - chatStage.clientHeight < 120;
}

function scrollToLatest(smooth = false) {
  chatStage.scrollTo({ top: chatStage.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  jumpLatest.hidden = true;
}

function revealUpdate() {
  if (isNearBottom()) scrollToLatest();
  else jumpLatest.hidden = false;
}

function syncControls() {
  const ownJob = Boolean(job && job.threadId === active.id);
  stopButton.hidden = !ownJob;
  sendButton.disabled = presence === "unsupported" || !promptBox.value.trim();
  document.body.classList.toggle("is-generating", Boolean(job));
  updatePresenceLine();
}

/* ---------- Conversation list ---------- */

function renderConversationList() {
  conversationList.replaceChildren();
  historyCount.textContent = String(threads.length);
  menuUnread.hidden = ![...unread].some((id) => id !== active.id);
  if (threads.length === 0) {
    conversationList.append(element("p", "empty-history", "Your chats will appear here."));
    return;
  }
  for (const item of threads) {
    const button = element("button", `conversation-item${item.id === active.id && !feedVisible ? " active" : ""}${unread.has(item.id) ? " unread" : ""}`);
    button.type = "button";
    button.dataset.conversationId = item.id;
    button.setAttribute("aria-current", item.id === active.id && !feedVisible ? "page" : "false");
    const top = element("span", "conversation-top");
    top.append(element("span", "conversation-title", item.title), element("span", "conversation-time", item.updatedAt ? shortWhen(item.updatedAt) : ""));
    const lastMessage = item.messages[item.messages.length - 1];
    let preview = "";
    if (job && job.threadId === item.id) preview = "typing…";
    else if (lastMessage?.role === "user") preview = `You: ${lastMessage.content}`;
    else if (lastMessage) preview = replyLines(lastMessage.content).join(" ");
    const bottom = element("span", "conversation-preview", preview.replace(/\s+/g, " "));
    if (job && job.threadId === item.id) bottom.classList.add("typing");
    button.append(top, bottom);
    if (unread.has(item.id)) button.setAttribute("aria-label", `${item.title}, new reply`);
    conversationList.append(button);
  }
}

/* ---------- Turn-taking and generation ---------- */

function schedulePump(delay = SETTLE_MS) {
  clearTimeout(settleTimer);
  settleTimer = setTimeout(() => { settleTimer = null; void pump(); }, delay);
}

async function pump() {
  if (job || !engine) return;
  for (const id of [...pending]) {
    const thread = findThread(id);
    if (!thread || !needsReply(thread)) pending.delete(id);
  }
  if (pending.size === 0) return;
  // The open chat goes first; others are answered in the background.
  const thread = pending.has(active.id) ? active : findThread([...pending][0]);
  if (!thread) return;
  await replyTo(thread);
  if (pending.size > 0) schedulePump(SETTLE_MS / 2);
}

function formatConversationLog(messages) {
  return messages.slice(-12).map(({ role, content, timestamp }) => {
    const date = new Date(timestamp || Date.now());
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hour = String(date.getHours()).padStart(2, "0");
    const minute = String(date.getMinutes()).padStart(2, "0");
    const speaker = role === "assistant" ? "自分" : "参加者A";
    return `${month}-${day} ${hour}:${minute} 【${speaker}】${content}`;
  }).join("\n");
}

const INTERNAL_HEADING = /(?:^|\n)\s*(?:\*{0,2}\s*)?(?:thinking process|chain of thought|analysis|reasoning)\s*:\s*/i;
const SELF_TALK = /^(?:let me think\b|i(?:'|’)ll think\b|let(?:'|’)s reason\b)/i;

// The part of a streaming reply that is safe to show, or null while it looks
// like hidden reasoning. Only finished lines are shown; the rest stays behind
// the typing indicator until toUserFacingReply() settles the final text.
function visibleSoFar(raw) {
  let text = raw.replace(/<think>[\s\S]*?<\/think>/gi, "");
  text = text.replace(/<think>[\s\S]*$/i, "").replace(/<\/?think>/gi, "").replace(/<[^>\n]*$/, "").replace(/^\s+/, "");
  if (INTERNAL_HEADING.test(text) || SELF_TALK.test(text)) return null;
  return text;
}

async function replyTo(thread) {
  const trailing = trailingUserMessages(thread);
  if (trailing.length === 0) return;
  const combined = trailing.map((message) => message.content).join("\n");
  const afterIndex = thread.messages.lastIndexOf(trailing[trailing.length - 1]);
  job = { threadId: thread.id, afterIndex, lines: [], stopped: false, startedAt: Date.now() };
  failures.delete(thread.id);
  let lease = null;
  let raw = "";
  try {
    lease = await engineScheduler.acquire("chat", { priority: true });
    const readAt = Date.now();
    for (const message of trailing) message.readAt = message.readAt || readAt;
    persistThreads();
    if (thread === active) renderThread();
    renderConversationList();
    liveStatus.textContent = "kobajinAI is typing.";

    const memory = lookupMemories(combined);
    const memoryContext = getMemoryContext(combined);
    const systemPrompt = memoryContext ? `${SYSTEM_PROMPT}\n\n${memoryContext}` : SYSTEM_PROMPT;
    const chunks = await engine.chat.completions.create({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: formatConversationLog(thread.messages.slice(0, afterIndex + 1)) },
      ],
      temperature: 0.72,
      top_p: 1,
      repetition_penalty: 1.12,
      max_tokens: 160,
      extra_body: { enable_thinking: false },
      stream: true,
    });
    for await (const chunk of chunks) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (typeof delta !== "string" || delta.length === 0) continue;
      raw += delta;
      if (job.stopped) continue;
      const visible = visibleSoFar(raw);
      if (visible === null) continue;
      const finished = visible.split("\n").slice(0, -1).map((line) => line.trim()).filter(Boolean);
      if (finished.length > job.lines.length) {
        job.lines = finished;
        if (thread === active) { renderLiveGroup(); revealUpdate(); }
      }
    }

    let reply;
    if (job.stopped) {
      const visible = (visibleSoFar(raw) || "").trim();
      reply = visible ? toUserFacingReply(visible, combined) : "";
    } else {
      reply = toUserFacingReply(raw, combined);
    }
    if (reply) {
      thread.messages.splice(afterIndex + 1, 0, { role: "assistant", content: reply, timestamp: job.startedAt, ...(memory.kind !== "none" ? { memory } : {}), ...(job.stopped ? { stopped: true } : {}) });
      if (thread !== active) unread.add(thread.id);
      // Its lines were already shown live; don't animate them in a second time.
      else if (rendered.threadId === thread.id) rendered.count += 1;
    }
    if (!reply || !needsReply(thread)) pending.delete(thread.id);
    job = null;
    // History may have been cleared while it was typing; don't bring the chat back.
    if (threads.includes(thread) || thread === active) saveThread(thread);
    liveStatus.textContent = reply ? "kobajinAI replied." : "Reply stopped.";
  } catch (error) {
    console.error("kobajinAI response generation failed", error);
    failures.set(thread.id, isJapanese(combined) ? "うまく返せませんでした。" : "It couldn't write a reply.");
    pending.delete(thread.id);
    job = null;
    liveStatus.textContent = "There was a problem generating a response.";
  } finally {
    lease?.release();
    job = null;
    if (thread === active) renderThread();
    renderConversationList();
    syncControls();
  }
}

function toUserFacingReply(output, userMessage) {
  let reply = output
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>[\s\S]*$/gi, "")
    .replace(/<\/?think>/gi, "")
    .trim();

  const finalAnswer = /(?:^|\n)\s*(?:\*{0,2}\s*)?(?:final answer|final response|answer)\s*:\s*/i.exec(reply);
  const internalHeading = INTERNAL_HEADING.exec(reply);
  if (internalHeading) {
    reply = finalAnswer && finalAnswer.index > internalHeading.index
      ? reply.slice(finalAnswer.index + finalAnswer[0].length)
      : "";
  } else if (SELF_TALK.test(reply)) {
    reply = "";
  }

  reply = reply.trim();
  if (reply) return reply;
  return isJapanese(userMessage)
    ? "ごめんなさい。短い返事を作れませんでした。もう一度お願いします。"
    : "Sorry, I couldn't produce a clear reply. Please try again.";
}

function sendMessage(text) {
  const value = text.trim();
  if (!value || presence === "unsupported") return;
  active.messages.push({ role: "user", content: value, timestamp: Date.now() });
  failures.delete(active.id);
  pending.add(active.id);
  saveThread(active);
  promptBox.value = "";
  resizePrompt();
  renderThread();
  scrollToLatest();
  if (!engine) void ensureModel();
  else if (!job) schedulePump(SETTLE_MS);
}

function stopReply() {
  if (!job) return;
  job.stopped = true;
  void engine?.interruptGenerate?.();
}

/* ---------- Views ---------- */

function openConversation(id) {
  const selected = threads.find((item) => item.id === id);
  if (!selected) return;
  showChat();
  active = selected;
  unread.delete(active.id);
  renderThread();
  scrollToLatest();
  renderConversationList();
  if (engine && !job) void engine.resetChat().catch(() => {});
  if (needsReply(active) && !failures.has(active.id)) {
    pending.add(active.id);
    if (engine) schedulePump(600);
  }
  closeSidebar();
  focusComposer();
}

function startNewChat() {
  showChat();
  active = newThread();
  renderThread();
  renderConversationList();
  if (engine && !job) void engine.resetChat().catch(() => {});
  promptBox.value = "";
  resizePrompt();
  closeSidebar();
  focusComposer();
}

function focusComposer() {
  // Opening the keyboard on phones would cover the conversation that was just opened.
  if (!window.matchMedia("(pointer: coarse)").matches) promptBox.focus();
}

function showChat() {
  if (!feedVisible) return;
  feedVisible = false;
  feedStage.hidden = true;
  chatStage.hidden = false;
  composerArea.hidden = false;
  feedToggle.classList.remove("active");
  chatsTab.classList.add("active");
  feedToggle.setAttribute("aria-selected", "false");
  chatsTab.setAttribute("aria-selected", "true");
  chatTitle.textContent = "kobajinAI";
  feed.setVisible(false);
  setPresence(presence);
  renderConversationList();
}

function showFeed() {
  feedVisible = true;
  chatStage.hidden = true;
  composerArea.hidden = true;
  feedStage.hidden = false;
  feedToggle.classList.add("active");
  chatsTab.classList.remove("active");
  feedToggle.setAttribute("aria-selected", "true");
  chatsTab.setAttribute("aria-selected", "false");
  chatTitle.textContent = "kobajinAI · Feed";
  feed.setVisible(true);
  renderConversationList();
  closeSidebar();
}

function openAbout() {
  if (!aboutDialog.open) aboutDialog.showModal();
}

function openSidebar() {
  sidebar.classList.add("open");
  sidebarScrim.classList.add("visible");
}

function closeSidebar() {
  sidebar.classList.remove("open");
  sidebarScrim.classList.remove("visible");
}

function resizePrompt() {
  promptBox.style.height = "auto";
  promptBox.style.height = `${Math.min(promptBox.scrollHeight, 180)}px`;
  sendButton.disabled = presence === "unsupported" || !promptBox.value.trim();
}

/* ---------- Events ---------- */

loadButton.addEventListener("click", () => { void ensureModel(); });
$("#presence").addEventListener("click", () => {
  if (feedVisible) { showChat(); return; }
  if (!engine && presence !== "loading" && presence !== "unsupported") void ensureModel();
  else openAbout();
});

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(promptBox.value);
});

promptBox.addEventListener("input", () => {
  resizePrompt();
  // Still typing a follow-up: keep waiting for the pause.
  if (settleTimer && !job && pending.has(active.id) && promptBox.value.trim()) schedulePump(SETTLE_MS);
});
promptBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});
stopButton.addEventListener("click", stopReply);

$("#new-chat").addEventListener("click", startNewChat);
feedToggle.addEventListener("click", showFeed);
chatsTab.addEventListener("click", () => { showChat(); renderThread(); });
conversationList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-conversation-id]");
  if (item) openConversation(item.dataset.conversationId);
});
$("#sidebar-toggle").addEventListener("click", openSidebar);
sidebarScrim.addEventListener("click", closeSidebar);
chatStage.addEventListener("scroll", () => { if (isNearBottom()) jumpLatest.hidden = true; }, { passive: true });
jumpLatest.addEventListener("click", () => scrollToLatest(true));
$("#about-trigger").addEventListener("click", openAbout);
$("#about-trigger-top").addEventListener("click", openAbout);
$("#about-close").addEventListener("click", () => aboutDialog.close());
$("#about-done").addEventListener("click", () => aboutDialog.close());
$("#clear-history").addEventListener("click", () => {
  if (!window.confirm("Clear all conversations saved in this browser? This cannot be undone.")) return;
  stopReply();
  threads = [];
  pending.clear();
  unread.clear();
  failures.clear();
  try { localStorage.removeItem(conversationStorageKey); } catch { /* The in-memory history is still cleared. */ }
  startNewChat();
  liveStatus.textContent = "Conversation history cleared.";
  aboutDialog.close();
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    promptBox.value = button.dataset.prompt || "";
    resizePrompt();
    promptBox.focus();
  });
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    startNewChat();
  }
  if (event.key === "Escape") closeSidebar();
});

/* ---------- Start ---------- */

const feed = initFeed({
  scheduler: engineScheduler,
  getEngine: () => engine,
  isChatBusy: () => Boolean(job) || pending.size > 0,
  modelId: MODEL_ID,
  announce: (text) => { liveStatus.textContent = text; },
});

// A chat left waiting for a reply (closed mid-reply, or sent before the model
// was ready) reopens if it is recent, and is answered once the model is online.
const resumable = threads[0];
if (resumable && needsReply(resumable) && Date.now() - resumable.updatedAt < RESUME_WINDOW_MS) {
  active = resumable;
  pending.add(active.id);
}

setPresence("offline", {
  line: "Offline · send a message to bring it online",
  title: "Not on this device yet",
  detail: "About 2.2 GiB, downloaded once and kept in this browser.",
  button: "Bring online",
});
renderConversationList();
renderThread();
scrollToLatest();

(async () => {
  if (!HAS_WEBGPU && !USE_MOCK_ENGINE) {
    markUnsupported();
    renderThread();
    return;
  }
  const appConfig = buildAppConfig();
  try {
    modelCached = USE_MOCK_ENGINE ? DEV_FLAGS.has("cached") : Boolean(appConfig && await hasModelInCache(MODEL_ID, appConfig));
  } catch {
    modelCached = false;
  }
  if (modelCached && presence === "offline") {
    // Already saved here: open it right away so kobajinAI is online by the time you've typed.
    setPresence("offline", { line: "Saved on this device · opening…", title: "Saved on this device", detail: "Opening the saved model…", button: "" });
    void ensureModel();
  } else {
    updateComposerHint();
  }
})();

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}
