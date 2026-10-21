const fs = require('fs');
const path = require('path');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research\\papers';
fs.mkdirSync(dir, { recursive: true });
const urls = [
  ['gyrodeblurnet_cvpr25.pdf', 'https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Gyro-based_Neural_Single_Image_Deblurring_CVPR_2025_paper.pdf'],
  ['gyrodeblurnet_supp.pdf', 'https://openaccess.thecvf.com/content/CVPR2025/supplemental/Yang_Gyro-based_Neural_Single_CVPR_2025_supplemental.pdf'],
  ['whyte_ijcv12_nonuniform.pdf', 'https://www.di.ens.fr/willow/pdfs/ijcv12.pdf'],
  ['whyte_cvpr10_nonuniform.pdf', 'https://www.di.ens.fr/willow/pdfs/cvpr10b.pdf'],
  ['fergus_siggraph06.pdf', 'https://www.di.ens.fr/willow/pdfs/siggraph06.pdf'],
  ['karpenko_siggraph11.pdf', 'https://www.di.ens.fr/willow/pdfs/siggraph11.pdf'],
  ['ieee_steve6d_manual.pdf', 'https://image-engineering.de/content/products/equipment/measurement_devices/steve-6d/downloads/STEVE-6D_manual.pdf'],
];
(async () => {
  for (const [name, u] of urls) {
    try {
      const r = await fetch(u, { redirect: 'follow', headers: { 'User-Agent': 'Mozilla/5.0' } });
      const ct = r.headers.get('content-type') || '';
      const b = Buffer.from(await r.arrayBuffer());
      if (r.status === 200 && b.slice(0, 4).toString() === '%PDF') {
        fs.writeFileSync(path.join(dir, name), b);
        console.log('OK  ', name, r.status, b.length, ct);
      } else {
        console.log('SKIP', name, r.status, ct, b.length, b.slice(0, 60).toString('latin1').replace(/\n/g, ' '));
      }
    } catch (e) { console.log('ERR ', name, e.message); }
  }
})();
