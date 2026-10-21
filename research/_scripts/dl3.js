const fs = require('fs');
const path = require('path');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research\\papers';
const urls = [
  ['whyte_ijcv12_supp.pdf', 'https://static-content.springer.com/esm/art%3A10.1007%2Fs11263-011-0502-7/MediaObjects/11263_2011_502_MOESM1_ESM.pdf'],
  ['mit_cameramotion.pdf', 'http://web.media.mit.edu/~roarkeh/cameramotion_RWH.pdf'],
  ['whyte_poster_cvpr10.pdf', 'https://www.di.ens.fr/willow/research/deblurring/poster_cvpr10.pdf'],
  ['hal_thesis_2012GRENT061.pdf', 'https://theses.hal.science/tel-01124089v1/preview/2012GRENT061.pdf'],
  ['steve6d_flyer.pdf', 'https://image-engineering.de/content/products/equipment/measurement_devices/steve-6d/downloads/STEVE-6D_flyer.pdf'],
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
