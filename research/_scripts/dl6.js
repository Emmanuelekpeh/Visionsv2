const fs = require('fs');
const path = require('path');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research\\papers';
const urls = [
  ['xiao2006_camera_motion_esr.pdf', 'https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2006/ICIS-0-736/33685.pdf'],
  ['bbc_rd_1971_28.pdf', 'http://downloads.bbc.co.uk/rd/pubs/reports/1971-28.pdf'],
  ['deepois_csvt.pdf', 'http://liushuaicheng.org/CSVT/DeepOIS.pdf'],
  ['ijomam_cobot_handtremor.pdf', 'http://ijomam.com/wp-content/uploads/2020/11/pag.-233-242_CONCEPTS-AND-ACHIEVEMENTS-EXPERIMENTAL-MODELS-CYBER-MIXMECHATRONICS-SYSTEMS-FOR-SMART-COBOTIC-PLATFORMS.pdf'],
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
