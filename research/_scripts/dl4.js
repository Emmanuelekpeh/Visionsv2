const fs = require('fs');
const path = require('path');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research\\papers';
const urls = [
  ['whyte_ijcv12_a.pdf', 'https://www.di.ens.fr/willow/research/deblurring/whyte_ijcv12.pdf'],
  ['whyte_ijcv12_b.pdf', 'https://www.di.ens.fr/willow/research/deblurring/pdfs/whyte_ijcv12.pdf'],
  ['whyte_ijcv12_c.pdf', 'https://www.di.ens.fr/willow/research/deblurring/nonuniform_deblurring.pdf'],
  ['whyte_ijcv12_d.pdf', 'https://www.robots.ox.ac.uk/~vgg/publications/2012/Whyte12/whyte12.pdf'],
  ['fergus06_a.pdf', 'https://www.di.ens.fr/willow/research/deblurring/fergus06.pdf'],
  ['fergus06_b.pdf', 'https://www.robots.ox.ac.uk/~vgg/publications/papers/fergus06.pdf'],
  ['karpenko11.pdf', 'https://www.di.ens.fr/willow/research/deblurring/karpenko11.pdf'],
  ['hu_gyro_cvpr21.pdf', 'https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_Deblurring_Text_Images_Using_Gyroscope_From_Mobile_Phones_CVPR_2021_paper.pdf'],
  ['rim_rsblur_cvpr22.pdf', 'https://openaccess.thecvf.com/content/CVPR2022/papers/Rim_Real-World_Blur_Super-Resolution_From_Realistic_Blur_Synthesis_CVPR_2022_paper.pdf'],
  ['mustaniemi_gyro_iccv19.pdf', 'https://openaccess.thecvf.com/content_ICCV_2019/papers/Mustaniemi_Gyroscope-Aided_Motion_Deblurring_in_Mobile_Imaging_ICCV_2019_paper.pdf'],
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
