const fs = require('fs');
const path = require('path');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research\\papers';
const urls = [
  ['mustaniemi_wacv19_deep.pdf', 'https://arxiv.org/pdf/1810.00986'],
  ['mustaniemi_iccv19_fast.pdf', 'https://arxiv.org/pdf/1904.01594'],
  ['hu_gyro_text_arxiv.pdf', 'https://arxiv.org/pdf/2104.00317'],
  ['rim_rsblur_arxiv.pdf', 'https://arxiv.org/pdf/2204.00515'],
  ['mustaniemi_gyro_deblur_2018.pdf', 'https://library.utia.cas.cz/separaty/2018/ZOI/mustaniemi-0490049.pdf'],
];
(async () => {
  for (const [name, u] of urls) {
    try {
      const r = await fetch(u, { redirect: 'follow', headers: { 'User-Agent': 'Mozilla/5.0' } });
      const ct = r.headers.get('content-type') || '';
      const b = Buffer.from(await r.arrayBuffer());
      if (r.status === 200 && b.slice(0, 4).toString() === '%PDF') {
        fs.writeFileSync(path.join(dir, name), b);
        console.log('OK  ', name, r.status, b.length);
      } else {
        console.log('SKIP', name, r.status, ct, b.length);
      }
    } catch (e) { console.log('ERR ', name, e.message); }
  }
})();
