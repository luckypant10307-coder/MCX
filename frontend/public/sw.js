// Service worker: shows push notifications when the site is closed, and opens the
// right page when one is tapped. Deliberately no offline caching of data: a stale
// signal shown as current would be worse than no page.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("push", (event) => {
  let msg = {};
  try { msg = event.data ? event.data.json() : {}; } catch (e) { msg = { body: event.data && event.data.text() }; }
  const loud = msg.kind === "new_alert" || msg.kind === "no_data";
  event.waitUntil(self.registration.showNotification(msg.title || "Gold Spread Monitor", {
    body: msg.body || "",
    icon: "/icon-192.png",
    badge: "/badge-72.png",
    tag: msg.tag || msg.kind || "gsm",          // a newer message of the same kind replaces the older one
    renotify: loud,
    requireInteraction: loud,                   // trade alerts stay until dismissed
    silent: msg.kind === "digest",
    data: { url: msg.url || "/" },
  }));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = new URL((event.notification.data && event.notification.data.url) || "/", self.location.origin).href;
  event.waitUntil(self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((wins) => {
    for (const w of wins) {
      if (w.url.startsWith(self.location.origin)) { w.focus(); return w.navigate(url); }
    }
    return self.clients.openWindow(url);
  }));
});
