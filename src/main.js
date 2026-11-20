import {
  CreateWebWorkerMLCEngine,
  hasModelInCache,
  prebuiltAppConfig,
} from "@mlc-ai/web-llm";
import { describeMemoryLookup, getMemoryContext } from "./memory.js";
import { createEngineScheduler } from "./agents.js";
import { initFeed } from "./feed_view.js";
import SYSTEM_PROMPT_SOURCE from "../self_twin/inference/system_prompt.txt?raw";

const BASE_MODEL_ID = "Qwen3.5-4B-q4f16_1-MLC";
const MODEL_REVISION = "a2534a20253d54cce925307fbd76086498b09a11";
const MODEL_ID = "kobajinAI-qwen35-4b-luna-v3-a2534a2-q4f16_1-MLC";
const MODEL_URL = import.meta.env.VITE_MODEL_URL || `https://huggingface.co/kobajin/kobajinAI/resolve/${MODEL_REVISION}/`;
const SYSTEM_PROMPT = SYSTEM_PROMPT_SOURCE.trim();
const RING_TICKS = 72;
const $ = (selector) => document.querySelector(selector);
const welcome = $("#welcome");
const messagesView = $("#messages");
const promptBox = $("#prompt");
const sendButton = $("#send");
const loadButton = $("#load-model");
const loadLabel = $("#load-label");
const engineStatus = $("#engine-status");
const topbarProgressBar = $("#topbar-progress-bar");
const core = $("#core");
const coreCaption = $("#core-caption");
const coreRing = $("#core-ring");
const presenceLabel = $("#presence-label");
const composerNote = $("#composer-note");
const dialogDevice = $("#dialog-device");
const liveStatus = $("#live-status");
const conversationList = $("#conversation-list");
const historyCount = $("#history-count");
const chatTitle = $("#chat-title");
const sidebar = $("#sidebar");
const sidebarScrim = $("#sidebar-scrim");
const aboutDialog = $("#about-dialog");
const chatStage = $(".chat-stage");
const composerArea = $(".composer-area");
const feedStage = $("#feed-stage");
const feedToggle = $("#feed-toggle");

let engine = null;
let enginePromise = null;
let worker = null;
let conversation = [];
let currentConversationId = null;
const conversationStorageKey = "kobajinAI.conversations.v1";
let savedConversations = readSavedConversations();
let generating = false;
// Enter during a reply queues the next message instead of dropping it.
let queuedSend = false;
let savedOnDevice = false;
let gpuLabel = "";
const loadState = { fetched: false, overall: 0 };
const storagePersistence = requestStoragePersistence();
// One engine serves chat and the feed; chat interrupts autonomous feed posts.
const engineScheduler = createEngineScheduler();
engineScheduler.setInterrupt(() => engine?.interruptGenerate());
let feedVisible = false;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
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
        messages: item.messages.filter((message) => message && ["user", "assistant"].includes(message.role) && typeof message.content === "string"),
      }))
      .filter((item) => item.messages.length > 0)
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, 40);
  } catch {
    return [];
  }
}

function saveCurrentConversation() {
  if (!conversation.some((message) => message.role === "user")) return;
  if (!currentConversationId) currentConversationId = globalThis.crypto?.randomUUID?.() || `chat-${Date.now()}`;
  const firstMessage = conversation.find((message) => message.role === "user")?.content || "New chat";
  const title = firstMessage.replace(/\s+/g, " ").trim().slice(0, 42) || "New chat";
  const saved = { id: currentConversationId, title, updatedAt: Date.now(), messages: conversation };
  savedConversations = [saved, ...savedConversations.filter((item) => item.id !== currentConversationId)]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 40);
  try {
    localStorage.setItem(conversationStorageKey, JSON.stringify(savedConversations));
  } catch {
    liveStatus.textContent = "This chat could not be saved in browser storage.";
  }
  chatTitle.textContent = title;
  renderConversationList();
}

function renderConversationList() {
  conversationList.replaceChildren();
  historyCount.textContent = String(savedConversations.length);
  if (savedConversations.length === 0) {
    conversationList.append(element("p", "empty-history", "Your chats will appear here."));
    return;
  }
  for (const item of savedConversations) {
    const button = document.createElement("button");
    button.className = `conversation-item${item.id === currentConversationId ? " active" : ""}`;
    button.type = "button";
    button.dataset.conversationId = item.id;
    button.setAttribute("aria-current", item.id === currentConversationId ? "page" : "false");
    button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H6l-3 2v-6.5A7.5 7.5 0 1 1 20 11.5Z" /></svg>';
    button.append(element("span", "", item.title));
    conversationList.append(button);
  }
}

async function requestStoragePersistence() {
  try {
    return await navigator.storage?.persist?.() ?? false;
  } catch {
    return false;
  }
}

/* ---------- Presence: one state for "where is the twin right now" ---------- */

function idlePresenceLabel() {
  if (engine) return "On this device";
  if (enginePromise) return `Arriving · ${Math.round(loadState.overall * 100)}%`;
  return savedOnDevice ? "Saved on this device" : "Not on this device";
}

// absent | arriving | ready | reading | typing | error
function setTwin(state, label = idlePresenceLabel()) {
  document.body.dataset.twin = state;
  presenceLabel.textContent = label;
}

function setStatus(kind, label) {
  engineStatus.className = `engine-status ${kind}`;
  engineStatus.querySelector("span").textContent = label;
}

// The welcome screen is only the mark. The caption under it speaks only when
// there is something to say: a hint on hover/focus, live loading, or an error.
function setCaption(text, kind = "hint") {
  coreCaption.textContent = text;
  coreCaption.dataset.kind = kind;
}

function idleCaption() {
  if (engine) return "";
  return savedOnDevice ? "Wake kobajinAI · saved in this browser" : "Bring kobajinAI here · ≈ 2.2 GiB, once";
}

function updateCore() {
  const label = engine
    ? "kobajinAI is here. Start typing."
    : enginePromise
      ? "kobajinAI is arriving on this device"
      : savedOnDevice
        ? "Wake kobajinAI. It is already saved in this browser."
        : "Bring kobajinAI to this device. About 2.2 GiB, downloaded once.";
  core.setAttribute("aria-label", label);
  if (!enginePromise) setCaption(idleCaption());
}

function buildRing() {
  const ns = "http://www.w3.org/2000/svg";
  coreRing.replaceChildren();
  for (let index = 0; index < RING_TICKS; index += 1) {
    const tick = document.createElementNS(ns, "line");
    const angle = (index / RING_TICKS) * Math.PI * 2 - Math.PI / 2;
    const inner = 86;
    const outer = 95;
    tick.setAttribute("x1", (100 + Math.cos(angle) * inner).toFixed(2));
    tick.setAttribute("y1", (100 + Math.sin(angle) * inner).toFixed(2));
    tick.setAttribute("x2", (100 + Math.cos(angle) * outer).toFixed(2));
    tick.setAttribute("y2", (100 + Math.sin(angle) * outer).toFixed(2));
    tick.style.setProperty("--i", String(index));
    coreRing.append(tick);
  }
}

function paintRing(fraction) {
  const lit = Math.round(Math.max(0, Math.min(1, fraction)) * RING_TICKS);
  coreRing.querySelectorAll("line").forEach((tick, index) => {
    tick.classList.toggle("lit", index < lit);
    tick.classList.toggle("head", index === lit - 1 && lit < RING_TICKS);
  });
}

function formatBytes(megabytes) {
  if (!Number.isFinite(megabytes)) return "";
  return megabytes >= 1024 ? `${(megabytes / 1024).toFixed(2)} GB` : `${megabytes} MB`;
}

let loadPhase = "Preparing";
let loadDetail = "";

function renderLoadCaption() {
  const percentage = Math.round(loadState.overall * 100);
  setCaption([`${loadPhase} ${percentage}%`, loadDetail].filter(Boolean).join(" · "), "progress");
}

function setOverall(value) {
  loadState.overall = Math.max(loadState.overall, Math.max(0, Math.min(1, value)));
  const percentage = Math.round(loadState.overall * 100);
  topbarProgressBar.style.width = `${percentage}%`;
  paintRing(loadState.overall);
  if (enginePromise && !engine) {
    presenceLabel.textContent = `Arriving · ${percentage}%`;
    renderLoadCaption();
  }
  messagesView.querySelectorAll(".turn.pending .turn-status").forEach((node) => {
    node.textContent = `Sends when kobajinAI arrives · ${percentage}%`;
  });
}

// WebLLM reports each phase from 0 to 1 again; map them onto one honest line.
function setLoadingProgress(progress) {
  const text = String(progress?.text || "");
  const fraction = Number.isFinite(progress?.progress) ? Math.max(0, Math.min(1, progress.progress)) : 0;
  const shard = /\[(\d+)\/(\d+)\]/.exec(text);
  const megabytes = /(\d+)MB/.exec(text);
  const size = megabytes ? formatBytes(Number(megabytes[1])) : "";

  if (text.startsWith("Fetching param cache")) {
    loadState.fetched = true;
    loadPhase = "Downloading";
    loadDetail = [shard ? `shard ${shard[1]}/${shard[2]}` : "", size].filter(Boolean).join(" · ");
    setOverall(fraction * 0.82);
  } else if (text.startsWith("Loading model from cache")) {
    loadPhase = "Loading onto GPU";
    loadDetail = shard ? `shard ${shard[1]}/${shard[2]}` : "";
    setOverall(loadState.fetched ? 0.82 + fraction * 0.08 : fraction * 0.9);
  } else if (text.startsWith("Loading GPU shader")) {
    loadPhase = "Compiling shaders";
    loadDetail = "";
    setOverall(0.9 + fraction * 0.09);
  } else if (text.startsWith("Finish loading on")) {
    gpuLabel = text.replace("Finish loading on", "").trim();
    setOverall(1);
  }
}

function buildAppConfig() {
  const baseRecord = prebuiltAppConfig.model_list.find((item) => item.model_id === BASE_MODEL_ID);
  if (!baseRecord) return null;
  return {
    ...prebuiltAppConfig,
    model_list: [...prebuiltAppConfig.model_list, { ...baseRecord, model_id: MODEL_ID, model: MODEL_URL }],
  };
}

function showArrivalError(detail) {
  topbarProgressBar.style.width = "0%";
  paintRing(0);
  setCaption(detail, "error");
}

function updateComposerNote() {
  if (engine) composerNote.textContent = "Generated on your device. It can be wrong about kobajin.";
  else if (enginePromise) composerNote.textContent = "You can type now — it sends when kobajinAI arrives.";
  else if (savedOnDevice) composerNote.textContent = "Saved in this browser — it wakes as you start typing.";
  else composerNote.textContent = "Your first message brings kobajinAI to this device (≈ 2.2 GiB, once).";
}

async function loadModel() {
  if (engine) return engine;
  if (enginePromise) return enginePromise;

  if (!navigator.gpu) {
    const message = "WebGPU is not available in this browser. Open this page in a recent desktop version of Chrome or Edge.";
    liveStatus.textContent = message;
    setStatus("error", "WebGPU unavailable");
    setTwin("error", "WebGPU unavailable");
    showArrivalError(message);
    return null;
  }

  const appConfig = buildAppConfig();
  if (!appConfig) {
    const message = `WebLLM ${BASE_MODEL_ID} runtime was not found. Refresh the app after updating its package.`;
    setStatus("error", "Model runtime unavailable");
    setTwin("error", "Runtime unavailable");
    showArrivalError(message);
    liveStatus.textContent = message;
    throw new Error(message);
  }

  loadState.fetched = false;
  loadState.overall = 0;
  loadPhase = savedOnDevice ? "Opening" : "Connecting";
  loadDetail = "";
  loadButton.disabled = true;
  loadLabel.textContent = "Preparing local model…";
  setStatus("loading", "Loading model");
  liveStatus.textContent = "Loading the kobajinAI model into this browser.";

  enginePromise = (async () => {
    try {
      savedOnDevice = await hasModelInCache(MODEL_ID, appConfig);
    } catch {
      // A cache lookup failure should not stop the normal WebLLM load path.
    }
    loadPhase = savedOnDevice ? "Opening" : "Downloading";
    renderLoadCaption();
    void storagePersistence;
    if (import.meta.env.DEV && new URLSearchParams(location.search).has("mock-engine")) {
      const { createMockEngine } = await import("./dev_mock_engine.js");
      engine = await createMockEngine(setLoadingProgress);
    } else {
      worker = new Worker(new URL("./webllm.worker.js", import.meta.url), { type: "module" });
      engine = await CreateWebWorkerMLCEngine(worker, MODEL_ID, {
        appConfig,
        initProgressCallback: setLoadingProgress,
      });
    }
    savedOnDevice = true;
    setOverall(1);
    setCaption("");
    if (gpuLabel) {
      dialogDevice.hidden = false;
      dialogDevice.textContent = `This session: ${gpuLabel} · WebGPU`;
      $("#presence").title = `Running on ${gpuLabel} through WebGPU`;
    }
    document.body.classList.add("twin-arrived");
    loadButton.classList.add("loaded");
    loadButton.disabled = true;
    loadLabel.textContent = "Model ready";
    loadButton.querySelector("svg").innerHTML = '<path d="m5 12 4 4L19 6" />';
    setStatus("ready", "Ready on this device");
    if (!generating) setTwin("ready");
    updateCore();
    updateComposerNote();
    liveStatus.textContent = gpuLabel ? `kobajinAI is ready, running on ${gpuLabel}.` : "kobajinAI is ready in this browser.";
    updateSendButton();
    promptBox.focus();
    feed.refresh();
    return engine;
  })().catch((error) => {
    console.error("kobajinAI model load failed", error);
    enginePromise = null;
    engine = null;
    worker?.terminate();
    worker = null;
    loadButton.disabled = false;
    loadLabel.textContent = "Try loading again";
    setStatus("error", "Could not load model");
    setTwin("error", "Could not load");
    updateCore();
    showArrivalError(`Couldn't load: ${error?.message || "check your connection and browser storage"}. Tap the mark to try again.`);
    updateComposerNote();
    liveStatus.textContent = "The local model could not be loaded.";
    throw error;
  });

  setTwin("arriving");
  updateCore();
  renderLoadCaption();
  updateComposerNote();
  return enginePromise;
}

/* ---------- Messages: kobajin's line breaks become separate bubbles ---------- */

function splitLines(text) {
  return String(text || "").split(/\n+/).map((line) => line.trim()).filter(Boolean);
}

// The streamed text before final cleanup; hides reasoning tags while they form.
function visibleStream(raw) {
  return raw
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>[\s\S]*$/i, "")
    .replace(/<\/?think>/gi, "")
    .replace(/<[^>\n]*$/, "")
    .replace(/^\s+/, "");
}

function nearBottom() {
  return chatStage.scrollHeight - chatStage.scrollTop - chatStage.clientHeight < 140;
}

function scrollToEnd(force = false) {
  if (force || nearBottom()) chatStage.scrollTop = chatStage.scrollHeight;
}

function showMessages() {
  messagesView.hidden = false;
  welcome.hidden = true;
}

function createTurn(role) {
  const turn = element("article", `turn ${role}`);
  if (role === "assistant") {
    const avatar = element("span", "turn-avatar");
    avatar.setAttribute("aria-hidden", "true");
    avatar.append(element("span", "mark"));
    turn.append(avatar);
  }
  const body = element("div", "turn-body");
  const bubbles = element("div", "bubbles");
  body.append(bubbles);
  turn.append(body);
  messagesView.append(turn);
  showMessages();
  return { turn, body, bubbles };
}

function setBubbles(container, lines) {
  const existing = [...container.querySelectorAll(".bubble:not(.ghost)")];
  lines.forEach((line, index) => {
    if (existing[index]) {
      if (existing[index].textContent !== line) existing[index].textContent = line;
    } else {
      container.append(element("div", "bubble", line));
    }
  });
  existing.slice(lines.length).forEach((node) => node.remove());
}

function formatSeconds(ms) {
  return ms < 10_000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms / 1000)} s`;
}

function formatObserved(value) {
  const match = /^(\d{4}-\d{2}-\d{2})/.exec(String(value || ""));
  return match ? match[1] : "";
}

function renderRecall(memory) {
  if (!memory || memory.state === "none") return null;
  const details = element("details", "recall");
  details.dataset.state = memory.state;
  const summary = element("summary");
  const count = memory.records?.length || 0;
  summary.append(
    element("span", "recall-icon"),
    element("span", "recall-label", memory.state === "found"
      ? `Recalled ${count} reviewed ${count === 1 ? "memory" : "memories"}`
      : "Looked for a memory · nothing confirmed"),
  );
  details.append(summary);
  const panel = element("div", "recall-panel");
  if (memory.state === "found") {
    for (const record of memory.records) {
      const item = element("div", "recall-item");
      item.append(element("p", "recall-content", record.content));
      const when = formatObserved(record.observed_at);
      const note = [when ? `Noted ${when}` : "", record.could_be_stale ? "may be out of date" : ""].filter(Boolean).join(" · ");
      if (note) item.append(element("p", "recall-note", note));
      panel.append(item);
    }
    panel.append(element("p", "recall-foot", "Offered to the model as context for this reply. It may or may not use them."));
  } else {
    panel.append(element("p", "recall-foot", "No reviewed memory matched this question, so kobajinAI was told not to guess personal details."));
  }
  details.append(panel);
  return details;
}

function renderMeta(message) {
  if (!Number.isFinite(message?.ms)) return null;
  return element("p", "turn-meta", `Generated on this device · ${formatSeconds(message.ms)}`);
}

function renderUserTurn(content) {
  const view = createTurn("user");
  view.bubbles.append(element("div", "bubble", content));
  return view;
}

function renderAssistantTurn(message) {
  const view = createTurn("assistant");
  const recall = renderRecall(message.memory);
  if (recall) view.body.insertBefore(recall, view.bubbles);
  setBubbles(view.bubbles, splitLines(message.content));
  const meta = renderMeta(message);
  if (meta) view.body.append(meta);
  return view;
}

function createLiveReply(memory) {
  const view = createTurn("assistant");
  view.turn.classList.add("live");
  const recall = renderRecall(memory);
  if (recall) view.body.insertBefore(recall, view.bubbles);
  const status = element("p", "turn-status");
  view.body.append(status);
  let ghost = null;
  let committed = 0;

  return {
    phase(name, label) {
      view.turn.dataset.phase = name;
      status.textContent = label;
      scrollToEnd();
    },
    stream(raw) {
      const parts = visibleStream(raw).split("\n");
      const partial = parts.pop() || "";
      const done = parts.map((line) => line.trim()).filter(Boolean);
      while (committed < done.length) {
        const bubble = element("div", "bubble arrived", done[committed]);
        view.bubbles.insertBefore(bubble, ghost);
        committed += 1;
      }
      if (partial.trim()) {
        if (!ghost) {
          ghost = element("div", "bubble ghost");
          ghost.setAttribute("aria-hidden", "true");
          view.bubbles.append(ghost);
        }
        ghost.textContent = partial.trim();
      } else if (ghost) {
        ghost.remove();
        ghost = null;
      }
      scrollToEnd();
    },
    finish(message) {
      ghost?.remove();
      ghost = null;
      setBubbles(view.bubbles, splitLines(message.content));
      status.remove();
      const meta = renderMeta(message);
      if (meta) view.body.append(meta);
      view.turn.classList.remove("live");
      delete view.turn.dataset.phase;
      scrollToEnd();
    },
    fail(text) {
      ghost?.remove();
      status.remove();
      view.bubbles.replaceChildren(element("div", "bubble message-error", text));
      view.turn.classList.remove("live");
      delete view.turn.dataset.phase;
    },
  };
}

function renderConversationMessages() {
  messagesView.replaceChildren();
  messagesView.hidden = true;
  welcome.hidden = true;
  for (const message of conversation) {
    if (message.role === "assistant") renderAssistantTurn(message);
    else renderUserTurn(message.content);
  }
  if (conversation.length === 0) welcome.hidden = false;
  scrollToEnd(true);
}

async function openConversation(id) {
  if (generating) return;
  const selected = savedConversations.find((item) => item.id === id);
  if (!selected) return;
  currentConversationId = selected.id;
  conversation = selected.messages.map((message) => ({ ...message }));
  showChat();
  chatTitle.textContent = selected.title;
  renderConversationMessages();
  renderConversationList();
  if (engine) await engine.resetChat().catch(() => {});
  closeSidebar();
  promptBox.focus();
}

function startNewChat() {
  if (generating) return;
  showChat();
  currentConversationId = null;
  conversation = [];
  messagesView.replaceChildren();
  messagesView.hidden = true;
  welcome.hidden = false;
  chatTitle.textContent = "New chat";
  renderConversationList();
  if (engine) void engine.resetChat().catch(() => {});
  promptBox.value = "";
  resizePrompt();
  closeSidebar();
  promptBox.focus();
}

function showChat() {
  if (!feedVisible) return;
  feedVisible = false;
  feedStage.hidden = true;
  chatStage.hidden = false;
  composerArea.hidden = false;
  feedToggle.classList.remove("active");
  chatTitle.textContent = savedConversations.find((item) => item.id === currentConversationId)?.title || "New chat";
  feed.setVisible(false);
}

function showFeed() {
  feedVisible = true;
  chatStage.hidden = true;
  composerArea.hidden = true;
  feedStage.hidden = false;
  feedToggle.classList.add("active");
  chatTitle.textContent = "Feed";
  feed.setVisible(true);
  closeSidebar();
}

function openAbout() {
  if (!aboutDialog.open) aboutDialog.showModal();
}

function closeSidebar() {
  sidebar.classList.remove("open");
  sidebar.classList.remove("expanded");
  sidebarScrim.classList.remove("visible");
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

function updateSendButton() {
  const hasText = Boolean(promptBox.value.trim());
  if (!hasText) queuedSend = false;
  sendButton.disabled = !hasText || queuedSend;
  sendButton.classList.toggle("queued", queuedSend);
  sendButton.setAttribute("aria-label", queuedSend
    ? "Queued — sends after this reply"
    : generating ? "Queue message for after this reply" : "Send message");
}

function setBusy(busy) {
  generating = busy;
  sendButton.classList.toggle("busy", busy);
  updateSendButton();
  document.body.classList.toggle("is-generating", busy);
}

async function sendMessage(text) {
  const value = text.trim();
  if (!value || generating) return;
  setBusy(true);
  let reply = null;
  let engineLease = null;
  const sentAt = Date.now();

  // Show the visitor's message right away; if the twin is still arriving it waits here.
  const userView = renderUserTurn(value);
  promptBox.value = "";
  resizePrompt();
  scrollToEnd(true);
  let model = engine;
  if (!model) {
    userView.turn.classList.add("pending");
    const waiting = element("p", "turn-status", "Sends when kobajinAI arrives");
    userView.body.append(waiting);
    try {
      model = await loadModel();
    } catch {
      model = null;
    }
    waiting.remove();
    userView.turn.classList.remove("pending");
    if (!model) {
      userView.turn.remove();
      if (messagesView.children.length === 0) {
        messagesView.hidden = true;
        welcome.hidden = false;
      }
      queuedSend = false;
      if (!promptBox.value.trim()) promptBox.value = value;
      setBusy(false);
      resizePrompt();
      return;
    }
  }

  try {
    conversation.push({ role: "user", content: value, timestamp: sentAt });
    saveCurrentConversation();
    const memory = describeMemoryLookup(value);
    reply = createLiveReply(memory);
    reply.phase("reading", memory.state === "found" ? "Recalled a memory · reading the conversation" : "Reading the conversation");
    setTwin("reading", "Reading…");
    engineLease = await engineScheduler.acquire("chat", { priority: true });
    const startedAt = performance.now();
    const memoryContext = getMemoryContext(value);
    const systemPrompt = memoryContext ? `${SYSTEM_PROMPT}\n\n${memoryContext}` : SYSTEM_PROMPT;
    const chunks = await model.chat.completions.create({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: formatConversationLog(conversation) },
      ],
      temperature: 0.72,
      top_p: 1,
      repetition_penalty: 1.12,
      max_tokens: 160,
      extra_body: { enable_thinking: false },
      stream: true,
    });
    let rawReply = "";
    let typing = false;
    for await (const chunk of chunks) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (typeof delta === "string" && delta.length > 0) {
        rawReply += delta;
        if (!typing && visibleStream(rawReply).trim()) {
          typing = true;
          reply.phase("typing", "");
          setTwin("typing", "Typing…");
        }
        reply.stream(rawReply);
      }
    }
    const content = toUserFacingReply(rawReply, value);
    const message = {
      role: "assistant",
      content,
      timestamp: Date.now(),
      ms: Math.round(performance.now() - startedAt),
    };
    if (memory.state !== "none") message.memory = memory;
    reply.finish(message);
    conversation.push(message);
    saveCurrentConversation();
    liveStatus.textContent = `kobajinAI: ${splitLines(content).join(" / ")}`;
  } catch (error) {
    console.error("kobajinAI response generation failed", error);
    reply?.fail(/[぀-ヿ㐀-鿿]/.test(value)
      ? "うまく返せませんでした。もう一度お試しください。"
      : "I couldn't generate a reply. Please try again.");
    liveStatus.textContent = "There was a problem generating a response.";
  } finally {
    engineLease?.release();
    setBusy(false);
    setTwin(engine ? "ready" : "absent");
    promptBox.focus();
    if (queuedSend && promptBox.value.trim()) {
      queuedSend = false;
      setTimeout(() => { void sendMessage(promptBox.value); }, 120);
    }
    updateSendButton();
  }
}

function resizePrompt() {
  promptBox.style.height = "auto";
  promptBox.style.height = `${Math.min(promptBox.scrollHeight, 180)}px`;
  updateSendButton();
  document.body.classList.toggle("is-composing", Boolean(promptBox.value.trim()) && !generating);
}

function toUserFacingReply(output, userMessage) {
  let reply = output
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>[\s\S]*$/gi, "")
    .replace(/<\/?think>/gi, "")
    .trim();

  const finalAnswer = /(?:^|\n)\s*(?:\*{0,2}\s*)?(?:final answer|final response|answer)\s*:\s*/i.exec(reply);
  const internalHeading = /(?:^|\n)\s*(?:\*{0,2}\s*)?(?:thinking process|chain of thought|analysis|reasoning)\s*:\s*/i.exec(reply);
  if (internalHeading) {
    reply = finalAnswer && finalAnswer.index > internalHeading.index
      ? reply.slice(finalAnswer.index + finalAnswer[0].length)
      : "";
  } else if (/^(?:let me think\b|i(?:'|’)ll think\b|let(?:'|’)s reason\b)/i.test(reply)) {
    reply = "";
  }

  reply = reply.trim();
  if (reply) return reply;
  return /[぀-ヿ㐀-鿿]/.test(userMessage)
    ? "ごめんなさい。短い返事を作れませんでした。もう一度お願いします。"
    : "Sorry, I couldn't produce a clear reply. Please try again.";
}

async function detectSavedModel() {
  if (!navigator.gpu) {
    setTwin("error", "WebGPU unavailable");
    setStatus("error", "WebGPU unavailable");
    setCaption("This browser has no WebGPU. Try a recent desktop Chrome or Edge.", "error");
    core.disabled = true;
    composerNote.textContent = "This browser can't run kobajinAI (no WebGPU).";
    return;
  }
  const appConfig = buildAppConfig();
  if (!appConfig) return;
  try {
    savedOnDevice = await hasModelInCache(MODEL_ID, appConfig);
  } catch {
    savedOnDevice = false;
  }
  if (savedOnDevice && !engine && !enginePromise) {
    loadLabel.textContent = "Open saved model";
    setTwin("absent");
    updateCore();
    updateComposerNote();
  }
}

loadButton.addEventListener("click", () => {
  loadModel().catch(() => {});
});
core.addEventListener("click", () => {
  if (engine) promptBox.focus();
  else loadModel().catch(() => {});
});

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  if (generating) {
    if (promptBox.value.trim()) {
      queuedSend = true;
      updateSendButton();
      liveStatus.textContent = "Queued. It sends after this reply.";
    }
    return;
  }
  void sendMessage(promptBox.value);
});

promptBox.addEventListener("input", () => {
  resizePrompt();
  // A model already saved in this browser costs no download, so start waking it
  // on the first keystroke; by the time the message is sent it is mostly ready.
  if (savedOnDevice && !engine && !enginePromise && promptBox.value.trim()) loadModel().catch(() => {});
});
promptBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});

$("#new-chat").addEventListener("click", startNewChat);
feedToggle.addEventListener("click", () => {
  if (feedVisible) showChat();
  else showFeed();
});
$("#composer-new-chat").addEventListener("click", startNewChat);
conversationList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-conversation-id]");
  if (item) void openConversation(item.dataset.conversationId);
});
$("#sidebar-toggle").addEventListener("click", () => {
  sidebar.classList.add("open");
  sidebarScrim.classList.add("visible");
});
$("#history-toggle").addEventListener("click", () => {
  if (window.matchMedia("(max-width: 760px)").matches) {
    sidebar.classList.add("open");
    sidebarScrim.classList.add("visible");
  } else {
    sidebar.classList.toggle("expanded");
  }
});
sidebarScrim.addEventListener("click", closeSidebar);
$("#about-trigger").addEventListener("click", openAbout);
$("#about-trigger-top").addEventListener("click", openAbout);
$("#about-close").addEventListener("click", () => aboutDialog.close());
$("#about-done").addEventListener("click", () => aboutDialog.close());
$("#clear-history").addEventListener("click", () => {
  if (!window.confirm("Clear all conversations saved in this browser? This cannot be undone.")) return;
  savedConversations = [];
  try { localStorage.removeItem(conversationStorageKey); } catch { /* The in-memory history is still cleared. */ }
  startNewChat();
  liveStatus.textContent = "Conversation history cleared.";
  aboutDialog.close();
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    startNewChat();
  }
  if (event.key === "Escape") closeSidebar();
});

buildRing();
paintRing(0);
renderConversationList();
updateCore();
updateComposerNote();
void detectSavedModel();

const feed = initFeed({
  scheduler: engineScheduler,
  getEngine: () => engine,
  isChatBusy: () => generating,
  modelId: MODEL_ID,
  announce: (text) => { liveStatus.textContent = text; },
});

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
  });
}
