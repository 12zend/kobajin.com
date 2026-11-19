import {
  CreateWebWorkerMLCEngine,
  hasModelInCache,
  prebuiltAppConfig,
} from "@mlc-ai/web-llm";
import { getMemoryContext } from "./memory.js";
import { createEngineScheduler } from "./agents.js";
import { initFeed } from "./feed_view.js";
import SYSTEM_PROMPT_SOURCE from "../self_twin/inference/system_prompt.txt?raw";

const BASE_MODEL_ID = "Qwen3.5-4B-q4f16_1-MLC";
const MODEL_REVISION = "a2534a20253d54cce925307fbd76086498b09a11";
const MODEL_ID = "kobajinAI-qwen35-4b-luna-v3-a2534a2-q4f16_1-MLC";
const MODEL_URL = import.meta.env.VITE_MODEL_URL || `https://huggingface.co/kobajin/kobajinAI/resolve/${MODEL_REVISION}/`;
const SYSTEM_PROMPT = SYSTEM_PROMPT_SOURCE.trim();
const $ = (selector) => document.querySelector(selector);
const welcome = $("#welcome");
const messagesView = $("#messages");
const promptBox = $("#prompt");
const sendButton = $("#send");
const loadButton = $("#load-model");
const loadLabel = $("#load-label");
const engineStatus = $("#engine-status");
const progressWrap = $("#progress-wrap");
const progressTitle = $("#progress-title");
const progressPercent = $("#progress-percent");
const progressBar = $("#progress-bar");
const progressDetail = $("#progress-detail");
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
const storagePersistence = requestStoragePersistence();
// One engine serves chat and the feed; chat interrupts autonomous feed posts.
const engineScheduler = createEngineScheduler();
engineScheduler.setInterrupt(() => engine?.interruptGenerate());
let feedVisible = false;

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
    const empty = document.createElement("p");
    empty.className = "empty-history";
    empty.textContent = "Your chats will appear here.";
    conversationList.append(empty);
    return;
  }
  for (const item of savedConversations) {
    const button = document.createElement("button");
    button.className = `conversation-item${item.id === currentConversationId ? " active" : ""}`;
    button.type = "button";
    button.dataset.conversationId = item.id;
    button.setAttribute("aria-current", item.id === currentConversationId ? "page" : "false");
    button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H6l-3 2v-6.5A7.5 7.5 0 1 1 20 11.5Z" /></svg>';
    const label = document.createElement("span");
    label.textContent = item.title;
    button.append(label);
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

function setStatus(kind, label) {
  engineStatus.className = `engine-status ${kind}`;
  engineStatus.querySelector("span").textContent = label;
}

function setLoadingProgress(progress) {
  const percentage = Number.isFinite(progress?.progress)
    ? Math.max(0, Math.min(100, Math.round(progress.progress * 100)))
    : 0;
  progressTitle.textContent = progress?.text || "Preparing your local session";
  progressPercent.textContent = `${percentage}%`;
  progressBar.style.width = `${percentage}%`;
  progressDetail.textContent = percentage >= 99
    ? "Warming up the model on this device…"
    : "The first download is large. Your browser will cache it for next time.";
}

async function loadModel() {
  if (engine) return engine;
  if (enginePromise) return enginePromise;

  if (!navigator.gpu) {
    const message = "WebGPU is not available in this browser. Open this page in a recent desktop version of Chrome or Edge.";
    liveStatus.textContent = message;
    setStatus("error", "WebGPU unavailable");
    progressWrap.hidden = false;
    progressTitle.textContent = "This browser cannot run the local model";
    progressPercent.textContent = "—";
    progressBar.style.width = "0%";
    progressDetail.textContent = message;
    return null;
  }

  const baseRecord = prebuiltAppConfig.model_list.find((item) => item.model_id === BASE_MODEL_ID);
  if (!baseRecord) {
    const message = `WebLLM ${BASE_MODEL_ID} runtime was not found. Refresh the app after updating its package.`;
    setStatus("error", "Model runtime unavailable");
    progressWrap.hidden = false;
    progressTitle.textContent = "This app version does not include the model runtime";
    progressPercent.textContent = "!";
    progressDetail.textContent = message;
    liveStatus.textContent = message;
    throw new Error(message);
  }

  progressWrap.hidden = false;
  loadButton.disabled = true;
  loadLabel.textContent = "Preparing local model…";
  setStatus("loading", "Loading model");
  progressTitle.textContent = "Connecting to your model repository";
  progressPercent.textContent = "0%";
  progressBar.style.width = "0%";
  liveStatus.textContent = "Loading the kobajinAI model into this browser.";

  const customRecord = {
    ...baseRecord,
    model_id: MODEL_ID,
    model: MODEL_URL,
  };
  const appConfig = {
    ...prebuiltAppConfig,
    model_list: [...prebuiltAppConfig.model_list, customRecord],
  };

  let modelIsCached = false;
  try {
    modelIsCached = await hasModelInCache(MODEL_ID, appConfig);
  } catch {
    // A cache lookup failure should not stop the normal WebLLM load path.
  }
  progressTitle.textContent = modelIsCached
    ? "Opening the model saved in this browser"
    : "Downloading the model for the first time · about 2.2 GiB";
  progressDetail.textContent = modelIsCached
    ? "The model is already saved here. Preparing it for this session…"
    : "The first download can take a while. The browser will keep it for next time.";
  void storagePersistence;

  enginePromise = (async () => {
    worker = new Worker(new URL("./webllm.worker.js", import.meta.url), { type: "module" });
    engine = await CreateWebWorkerMLCEngine(worker, MODEL_ID, {
      appConfig,
      initProgressCallback: setLoadingProgress,
    });
    progressWrap.hidden = true;
    loadButton.classList.add("loaded");
    loadButton.disabled = true;
    loadLabel.textContent = "Model ready";
    loadButton.querySelector("svg").innerHTML = '<path d="m5 12 4 4L19 6" />';
    setStatus("ready", "Ready on this device");
    liveStatus.textContent = "kobajinAI is ready in this browser.";
    sendButton.disabled = !promptBox.value.trim();
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
    progressTitle.textContent = "The model could not be loaded";
    progressPercent.textContent = "!";
    progressBar.style.width = "0%";
    progressDetail.textContent = error?.message || "Check your connection and available browser storage, then try again.";
    liveStatus.textContent = "The local model could not be loaded.";
    throw error;
  });

  return enginePromise;
}

function appendMessage(role, content = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const body = document.createElement("div");
  body.className = "message-body";
  const text = document.createElement("div");
  text.className = "message-text";
  text.textContent = content;
  body.append(text);
  article.append(body);
  messagesView.append(article);
  messagesView.hidden = false;
  welcome.hidden = true;
  messagesView.scrollTop = messagesView.scrollHeight;
  return text;
}

function renderConversationMessages() {
  messagesView.replaceChildren();
  messagesView.hidden = true;
  welcome.hidden = true;
  for (const message of conversation) appendMessage(message.role, message.content);
  if (conversation.length === 0) welcome.hidden = false;
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

function setBusy(busy) {
  generating = busy;
  sendButton.disabled = busy || !promptBox.value.trim();
  sendButton.classList.toggle("busy", busy);
  sendButton.setAttribute("aria-label", busy ? "Generating response" : "Send message");
  promptBox.disabled = busy;
  document.body.classList.toggle("is-generating", busy);
}

async function sendMessage(text) {
  const value = text.trim();
  if (!value || generating) return;
  setBusy(true);
  let responseView = null;
  let engineLease = null;

  try {
    const model = await loadModel();
    if (!model) return;
    conversation.push({ role: "user", content: value, timestamp: Date.now() });
    saveCurrentConversation();
    appendMessage("user", value);
    promptBox.value = "";
    resizePrompt();
    responseView = appendMessage("assistant", "");
    responseView.classList.add("streaming");
    engineLease = await engineScheduler.acquire("chat", { priority: true });
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
    for await (const chunk of chunks) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (typeof delta === "string" && delta.length > 0) {
        rawReply += delta;
      }
    }
    const reply = toUserFacingReply(rawReply, value);
    responseView.classList.remove("streaming");
    responseView.textContent = reply;
    conversation.push({ role: "assistant", content: reply, timestamp: Date.now() });
    saveCurrentConversation();
    liveStatus.textContent = "Response complete.";
  } catch (error) {
    console.error("kobajinAI response generation failed", error);
    if (responseView) {
      responseView.classList.remove("streaming");
      responseView.classList.add("message-error");
      responseView.textContent = /[\u3040-\u30ff\u3400-\u9fff]/.test(value)
        ? "うまく返せませんでした。もう一度お試しください。"
        : "I couldn't generate a reply. Please try again.";
    }
    liveStatus.textContent = "There was a problem generating a response.";
  } finally {
    engineLease?.release();
    setBusy(false);
    promptBox.focus();
    sendButton.disabled = !promptBox.value.trim();
  }
}

function resizePrompt() {
  promptBox.style.height = "auto";
  promptBox.style.height = `${Math.min(promptBox.scrollHeight, 180)}px`;
  sendButton.disabled = generating || !promptBox.value.trim();
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
  return /[\u3040-\u30ff\u3400-\u9fff]/.test(userMessage)
    ? "ごめんなさい。短い返事を作れませんでした。もう一度お願いします。"
    : "Sorry, I couldn't produce a clear reply. Please try again.";
}

loadButton.addEventListener("click", () => {
  loadModel().catch(() => {});
});

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  void sendMessage(promptBox.value);
});

promptBox.addEventListener("input", resizePrompt);
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

renderConversationList();

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
