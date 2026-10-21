import pymupdf, json, math, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
path = Path(__file__).parent.parent / "_shake_research" / "CIPA.pdf"
doc = pymupdf.open(path)
for pno in (31, 32):  # 0-based -> printed pages 32,33 (Fig 4-5-1c, 4-5-1d)
    p = doc[pno]
    print('===== PDF page %d rect %s' % (pno + 1, p.rect))
    print('--- TEXT ---')
    print(p.get_text())
    print('--- DRAWINGS (vector paths) ---')
    d = p.get_drawings()
    print('num paths', len(d))
    for i, path in enumerate(d[:80]):
        r = path['rect']
        print(i, 'type', path['type'], 'rect', [round(v, 1) for v in (r.x0, r.y0, r.x1, r.y1)], 'items', len(path['items']))
