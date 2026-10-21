const fs = require('fs');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research\\papers';
(async () => {
  for (const [n, u] of [['cm824_patent_us20200084358.pdf', 'https://patentimages.storage.googleapis.com/4f/51/17/555d3c03fea479/US20200084358A1.pdf']]) {
    try {
      const r = await fetch(u, { headers: { 'User-Agent': 'Mozilla/5.0' } });
      const b = Buffer.from(await r.arrayBuffer());
      fs.writeFileSync(dir + '\\' + n, b);
      console.log('OK', n, r.status, b.length);
    } catch (e) { console.log('ERR', n, e.message); }
  }
})();
