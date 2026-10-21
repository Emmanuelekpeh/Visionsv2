import pymupdf, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
path = Path(__file__).parent.parent / "_shake_research" / "CIPA.pdf"
doc = pymupdf.open(path)

# axis calibration from gridline path rects (page 32/33):
# y=394.2 -> 0.0 deg ; y=145.5 -> 3.5 deg  => 1 deg = 71.06 pt
Y0, YDEG = 394.2, (394.2 - 145.5) / 3.5
# x gridlines (shutter speeds): read from text positions instead
for pno in (31, 32):
    p = doc[pno]
    print('===== page', pno + 1)
    # x tick label positions
    for w in p.get_text('words'):
        t = w[4]
        if '/' in t and any(c.isdigit() for c in t):
            print('  xtick %-8s x0=%.1f x1=%.1f y=%.1f' % (t, w[0], w[2], w[1]))
    print('  --- curve paths ---')
    for i, path in enumerate(p.get_drawings()):
        items = path['items']
        if len(items) < 6:
            continue
        pts = []
        for it in items:
            if it[0] == 'l':
                pts.append((it[1].x, it[1].y))
                pts.append((it[2].x, it[2].y))
            elif it[0] == 'c':
                for q in it[1:]:
                    pts.append((q.x, q.y))
        if len(pts) < 6:
            continue
        xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
        # only keep paths that look like data curves (span most of the plot box)
        if max(xs) - min(xs) < 200:
            continue
        print('  PATH %d  npts=%d  x[%.1f..%.1f] y[%.1f..%.1f]' % (i, len(pts), min(xs), max(xs), min(ys), max(ys)))
        for x, y in pts:
            print('     x=%.2f y=%.2f  -> deg=%.4f' % (x, y, (Y0 - y) / YDEG))
