const fs = require('fs');
(async () => {
  const r = await fetch('https://www.cipa.jp/e/std/image-stabilization.html');
  const t = await r.text();
  const links = [...t.matchAll(/href="([^"]+)"/g)].map(m => m[1]);
  console.log([...new Set(links)].join('\n'));
  const txt = t.replace(/<script[\s\S]*?<\/script>/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
  console.log('--- TEXT ---');
  console.log(txt.slice(0, 3000));
})();
