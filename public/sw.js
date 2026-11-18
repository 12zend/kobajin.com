const SHELL_CACHE = "kobajin-shell-v3";
const SHELL_FILES = ["/", "/logo.svg", "/manifest.webmanifest"];

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    await cache.addAll(SHELL_FILES);
    try {
      const manifestResponse = await fetch("/build-manifest.json");
      if (manifestResponse.ok) {
        const manifest = await manifestResponse.json();
        const assets = Object.values(manifest).flatMap((entry) => [
          entry.file,
          ...(entry.css || []),
          ...(entry.assets || []),
        ]).filter(Boolean).map((file) => `/${file.replace(/^\//, "")}`);
        await cache.addAll([...new Set(assets)]);
      }
    } catch {
      // The shell remains available; fetched same-origin assets are cached as they are used.
    }
  })());
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key.startsWith("kobajin-shell-") && key !== SHELL_CACHE).map((key) => caches.delete(key))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request).then((response) => {
        if (response.ok) {
          const copy = response.clone();
          void caches.open(SHELL_CACHE).then((cache) => cache.put(request, copy));
        }
        return response;
      }).catch(async () => {
        if (request.mode === "navigate") return (await caches.match("/")) || Response.error();
        return Response.error();
      });
    })
  );
});
