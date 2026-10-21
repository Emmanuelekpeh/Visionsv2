const fs = require('fs');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research\\papers';
const urls = [
  ['golik_thesis.pdf', 'https://www.image-engineering.de/content/library/diploma_thesis/borys_golik_image_stabilizers.pdf'],
  ['golik_thesis2.pdf', 'https://www.image-engineering.de/index.php?eID=dumpFile&t=f&f=1901&token=1c0e0a48a8b6b1f2d1c08e0a2b1c6d5e'],
  ['jstage_ncrtv_23_377.pdf', 'https://www.jstage.jst.go.jp/article/ncrtv/23/0/23_377/_pdf/-char/ja'],
  ['ovren_icra15.pdf', 'https://liu.diva-portal.org/smash/get/diva2:841497/FULLTEXT01.pdf'],
  ['korea_handshake_camcorder.pdf', 'http://koreascience.or.kr/article/CFKO199811919577244.pdf'],
];
(async () => {
  for (const [name, u] of urls) {
    try {
      const r = await fetch(u, { redirect: 'follow', headers: { 'User-Agent': 'Mozilla/5.0' } });
      const ct = r.headers.get('content-type') || '';
      const b = Buffer.from(await r.arrayBuffer());
      if (r.status === 200 && b.slice(0, 4).toString() === '%PDF') {
        fs.writeFileSync(dir + '\\' + name, b);
        console.log('OK  ', name, r.status, b.length);
      } else { console.log('SKIP', name, r.status, ct, b.length); }
    } catch (e) { console.log('ERR ', name, e.message); }
  }
})();
