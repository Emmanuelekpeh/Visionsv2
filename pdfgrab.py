import sys, os, io, re, urllib.request, ssl
import pymupdf

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

def grab(url, out=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    data = urllib.request.urlopen(req, timeout=90, context=ctx).read()
    if out is None:
        out = os.path.join("pdfs", re.sub(r'[^A-Za-z0-9._-]', '_', url)[-80:] + ".pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)
    return out

def text(path, maxpages=None):
    d = pymupdf.open(path)
    n = d.page_count if maxpages is None else min(maxpages, d.page_count)
    parts = []
    for i in range(n):
        parts.append(f"\n===== PAGE {i+1} =====\n" + d[i].get_text())
    return "".join(parts)

if __name__ == "__main__":
    url = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    p = grab(url, out)
    t = text(p)
    print(f"### FILE: {p}  ({len(t)} chars)")
    print(t)
