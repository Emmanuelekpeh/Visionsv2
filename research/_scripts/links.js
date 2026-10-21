const fs = require('fs');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research';
(async () => {
  for (const u of ['https://www.cipa.jp/e/std/image-stabilization.html',
                   'https://www.cipa.jp/e/std/image-stabilization/index.html']) {
    try {
      const r = await fetch(u);
      const t = await r.text();
      console.log('===', u, r.status, t.length);
      const links = [...t.matchAll(/href="([^"]+)"/g)].map(m => m[1])
        .filter(h => /wave|wb-|zip|txt|documents/i.test(h));
      console.log([...new Set(links)].join('\n'));
    } catch (e) { console.log('ERR', u, e.message); }
  }
})();
