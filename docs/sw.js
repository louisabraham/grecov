const CACHE = "grecov-__CACHE_VERSION__";
const PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

// Assets to pre-cache on install (small, critical)
const PRECACHE = ["./", "./index.html", "./wheel.txt"];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  // Remove old caches
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = e.request.url;

  // Only intercept Pyodide CDN and same-origin requests
  if (!url.startsWith(PYODIDE_CDN) && new URL(url).origin !== self.location.origin) {
    return; // let the browser handle it
  }

  e.respondWith(
    caches.open(CACHE).then((cache) =>
      cache.match(e.request).then((cached) => {
        if (cached) return cached;
        return fetch(e.request).then((resp) => {
          // Cache successful responses and opaque responses (cross-origin no-cors)
          if (resp.ok || resp.type === "opaque") cache.put(e.request, resp.clone());
          return resp;
        });
      })
    )
  );
});
