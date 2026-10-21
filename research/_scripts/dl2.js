const fs = require('fs');
const path = require('path');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research\\papers';
const urls = [
  ['golik_diploma_stabilizers.pdf', 'https://www.image-engineering.de/content/library/diploma_thesis/borys_golik_image_stabilizers.pdf'],
  ['jstage_multiaxis_shake_ja.pdf', 'https://www.jstage.jst.go.jp/article/itetr/34.1/0/34.1_39/_pdf/-char/ja'],
  ['steve6d_manual.pdf', 'https://image-engineering.de/content/products/equipment/measurement_devices/steve-6d/downloads/STEVE-6D_manual.pdf'],
  ['whyte_ijcv12.pdf', 'https://www.di.ens.fr/willow/research/deblurring/whyte_ijcv12.pdf'],
  ['whyte_cvpr10.pdf', 'https://www.di.ens.fr/willow/research/deblurring/whyte_cvpr10.pdf'],
  ['fergus06.pdf', 'https://www.di.ens.fr/willow/research/deblurring/fergus06.pdf'],
  ['gyro_video_stab_arxiv.pdf', 'https://arxiv.org/pdf/2404.00916'],
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
        console.log('SKIP', name, r.status, ct, b.length, b.slice(0, 50).toString('latin1').replace(/\n/g, ' '));
      }
    } catch (e) { console.log('ERR ', name, e.message); }
  }
})();
