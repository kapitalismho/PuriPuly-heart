const repository = 'https://github.com/kapitalismho/PuriPuly-heart';

const headers = {
  'Content-Type': 'text/html; charset=utf-8',
  'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
  'Referrer-Policy': 'no-referrer',
  'X-Content-Type-Options': 'nosniff',
};

export function createServicePage(name: string, purpose: string) {
  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PuriPuly Heart — ${name}</title>
  <style>
    body { background: #f6f8fb; color: #19263a; font: 1.125rem/1.6 system-ui, sans-serif; }
    main { max-width: 42rem; margin: clamp(2rem, 10vh, 7rem) auto; padding: 0 1.5rem; }
    .project { color: #40546e; font-weight: 650; margin-bottom: 0.25rem; }
    h1 { font-size: clamp(2.25rem, 6vw, 3.5rem); line-height: 1.1; margin: 0 0 1.25rem; }
    ul { display: flex; flex-wrap: wrap; gap: 0.5rem 1.25rem; list-style: none; padding: 0; }
    nav { border-top: 1px solid #c6cfdb; margin-top: 2rem; }
    a { color: #17467e; text-underline-offset: 0.2em; }
  </style>
</head>
<body>
  <main>
    <p class="project">PuriPuly Heart</p>
    <h1>${name}</h1>
    <p>${purpose}</p>
    <p>PuriPuly Heart is an open-source desktop app for real-time speech recognition, translation, and captions in VRChat and other apps.</p>
    <nav aria-label="Project resources">
      <ul>
        <li><a href="${repository}">Official repository</a></li>
        <li><a href="${repository}/blob/dev/README.md#usage">Documentation</a></li>
        <li><a href="${repository}/blob/dev/PRIVACY.md">Privacy policy</a></li>
        <li><a href="${repository}/issues">Support</a></li>
      </ul>
    </nav>
  </main>
</body>
</html>`;

  return (url: URL, method: string): Response | undefined => {
    if (url.pathname !== '/' || url.protocol !== 'https:' || (method !== 'GET' && method !== 'HEAD')) {
      return undefined;
    }
    return new Response(method === 'HEAD' ? null : html, { headers });
  };
}
