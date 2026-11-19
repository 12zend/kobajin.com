// Local feed storage for kobajinAI posts, reply threads, and generation logs.
// IndexedDB first, localStorage as a fallback. Nothing here leaves the browser.
//
// post: { id, text, createdAt, mode: "auto"|"manual",
//         source: { title, url, excerpt, license, revid },
//         replies: [{ role: "user"|"assistant", content, timestamp }],
//         log: { url, seed, verdicts, retries, model_id, json_mode } }
// log entries record every pipeline run (accepted or not) for gate pass rates.
// They hold only the seed, final drafts, and gate verdicts – never hidden reasoning.

const DB_NAME = "kobajinAI.feed";
const DB_VERSION = 1;
const POSTS = "posts";
const LOGS = "logs";
const LOCAL_POSTS_KEY = "kobajinAI.feed.posts.v1";
const LOCAL_LOGS_KEY = "kobajinAI.feed.logs.v1";
const MAX_POSTS = 200;
const MAX_LOGS = 500;

function newId(prefix) {
  return `${prefix}-${globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
}

export function validatePost(post) {
  if (!post || typeof post.text !== "string" || !post.text.trim()) throw new Error("A feed post needs text");
  const url = post.source?.url;
  if (typeof url !== "string" || !/^https:\/\/ja\.wikipedia\.org\//.test(url)) {
    throw new Error("A feed post needs its Wikipedia source URL");
  }
  if (typeof post.source.title !== "string" || !post.source.title) throw new Error("A feed post needs its source title");
  return post;
}

function sanitizeReplies(replies) {
  return (Array.isArray(replies) ? replies : [])
    .filter((reply) => reply && ["user", "assistant"].includes(reply.role) && typeof reply.content === "string")
    .map(({ role, content, timestamp }) => ({ role, content, timestamp: Number(timestamp) || Date.now() }));
}

export function createPost({ text, article, log, mode }) {
  return validatePost({
    id: newId("post"),
    text: text.trim(),
    createdAt: Date.now(),
    mode,
    source: {
      title: article.title,
      url: article.url,
      excerpt: article.excerpt,
      license: article.license,
      revid: article.revid ?? null,
    },
    replies: [],
    log: {
      url: log.url,
      seed: log.seed,
      verdicts: log.verdicts,
      retries: log.retries,
      model_id: log.model_id,
      json_mode: log.json_mode,
    },
  });
}

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(POSTS)) db.createObjectStore(POSTS, { keyPath: "id" }).createIndex("createdAt", "createdAt");
      if (!db.objectStoreNames.contains(LOGS)) db.createObjectStore(LOGS, { keyPath: "id" }).createIndex("createdAt", "createdAt");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error("Feed database is blocked by another tab"));
  });
}

function indexedDbStore(db) {
  const tx = (store, mode = "readonly") => db.transaction(store, mode).objectStore(store);
  const newestFirst = async (store, limit) => {
    const rows = await requestResult(tx(store).index("createdAt").getAll());
    return rows.sort((a, b) => b.createdAt - a.createdAt).slice(0, limit);
  };
  const prune = async (store, keep) => {
    const rows = await newestFirst(store, Number.MAX_SAFE_INTEGER);
    const objectStore = tx(store, "readwrite");
    for (const row of rows.slice(keep)) objectStore.delete(row.id);
  };
  return {
    kind: "indexeddb",
    listPosts: (limit = MAX_POSTS) => newestFirst(POSTS, limit),
    async addPost(post) {
      await requestResult(tx(POSTS, "readwrite").put(validatePost(post)));
      await prune(POSTS, MAX_POSTS);
      return post;
    },
    async updateReplies(postId, replies) {
      const store = tx(POSTS, "readwrite");
      const post = await requestResult(store.get(postId));
      if (!post) return null;
      post.replies = sanitizeReplies(replies);
      await requestResult(store.put(post));
      return post;
    },
    async addLog(entry) {
      const row = { id: newId("log"), createdAt: Date.now(), ...entry };
      await requestResult(tx(LOGS, "readwrite").put(row));
      await prune(LOGS, MAX_LOGS);
      return row;
    },
    listLogs: (limit = MAX_LOGS) => newestFirst(LOGS, limit),
    async clear() {
      await requestResult(tx(POSTS, "readwrite").clear());
      await requestResult(tx(LOGS, "readwrite").clear());
    },
  };
}

function localStore() {
  const read = (key) => {
    try {
      const value = JSON.parse(localStorage.getItem(key) || "[]");
      return Array.isArray(value) ? value : [];
    } catch {
      return [];
    }
  };
  const write = (key, rows, keep) => {
    const sorted = rows.sort((a, b) => b.createdAt - a.createdAt).slice(0, keep);
    try { localStorage.setItem(key, JSON.stringify(sorted)); } catch { /* storage full: keep in memory only */ }
    return sorted;
  };
  return {
    kind: "localstorage",
    listPosts: async (limit = MAX_POSTS) => read(LOCAL_POSTS_KEY).slice(0, limit),
    async addPost(post) {
      write(LOCAL_POSTS_KEY, [validatePost(post), ...read(LOCAL_POSTS_KEY)], MAX_POSTS);
      return post;
    },
    async updateReplies(postId, replies) {
      const rows = read(LOCAL_POSTS_KEY);
      const post = rows.find((row) => row.id === postId);
      if (!post) return null;
      post.replies = sanitizeReplies(replies);
      write(LOCAL_POSTS_KEY, rows, MAX_POSTS);
      return post;
    },
    async addLog(entry) {
      const row = { id: newId("log"), createdAt: Date.now(), ...entry };
      write(LOCAL_LOGS_KEY, [row, ...read(LOCAL_LOGS_KEY)], MAX_LOGS);
      return row;
    },
    listLogs: async (limit = MAX_LOGS) => read(LOCAL_LOGS_KEY).slice(0, limit),
    async clear() {
      try {
        localStorage.removeItem(LOCAL_POSTS_KEY);
        localStorage.removeItem(LOCAL_LOGS_KEY);
      } catch { /* nothing else to clear */ }
    },
  };
}

export async function openFeedStore() {
  if (globalThis.indexedDB) {
    try {
      return indexedDbStore(await openDatabase());
    } catch (error) {
      console.warn("kobajinAI feed: IndexedDB unavailable, using localStorage", error);
    }
  }
  return localStore();
}

/** Gate pass rate over stored pipeline runs. */
export function summarizeLogs(logs) {
  const runs = logs.filter((log) => !log.interrupted);
  const accepted = runs.filter((log) => log.accepted).length;
  const attempts = runs.reduce((total, log) => total + (log.verdicts?.length || 0), 0);
  return {
    runs: runs.length,
    accepted,
    passRate: runs.length ? accepted / runs.length : 0,
    attempts,
    attemptPassRate: attempts ? accepted / attempts : 0,
  };
}
