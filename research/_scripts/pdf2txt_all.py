import pymupdf, sys, os, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
d = Path(__file__).parent.parent / "_shake_research" / "papers"
for f in sorted(d.iterdir()):
    if not f.suffix.lower() == '.pdf':
        continue
    src = f
    out = f.with_suffix('.txt')
    try:
        doc = pymupdf.open(src)
        t = '\n'.join('=== PAGE %d ===\n%s' % (i + 1, p.get_text()) for i, p in enumerate(doc))
        out.write_text(t, encoding='utf-8')
        print('%-40s pages=%-4d chars=%d' % (f.name, doc.page_count, len(t)))
    except Exception as e:
        print('ERR', f, e)
