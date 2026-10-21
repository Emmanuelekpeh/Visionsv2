const fs = require('fs');
const path = require('path');
const dir = 'C:\\Users\\emman\\Downloads\\visions\\_shake_research\\papers';
const urls = [
  ['steve6d_data_sheet.pdf', 'https://www.image-engineering.de/content/products/equipment/measurement_devices/steve-6d/downloads/STEVE-6DS_data_sheet.pdf'],
  ['steve_data_sheet.pdf', 'https://www.image-engineering.de/content/products/equipment/measurement_devices/steve/downloads/STEVE_data_sheet.pdf'],
  ['sensors_portal_aug2013.pdf', 'https://www.sensorsportal.com/DOWNLOADS/Journals/2013/ST_August_2013.pdf'],
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
