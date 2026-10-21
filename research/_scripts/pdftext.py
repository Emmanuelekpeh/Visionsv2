"""Throwaway PDF text extractor for research note-taking only (not a deliverable).
Usage: python pdftext.py file.pdf [outfile.txt]
"""
import re, sys, zlib

def get_streams(data):
    out = []
    for m in re.finditer(rb"stream\r?\n", data):
        start = m.end()
        e = data.find(b"endstream", start)
        if e == -1:
            continue
        raw = data[start:e]
        raw = raw.rstrip(b"\r\n")
        try:
            out.append(zlib.decompress(raw))
        except Exception:
            try:
                out.append(zlib.decompressobj().decompress(raw))
            except Exception:
                pass
    return out

def tounicode_maps(data):
    """Collect ToUnicode CMaps: code -> unicode char."""
    maps = []
    for s in get_streams(data):
        if b"beginbfchar" not in s and b"beginbfrange" not in s:
            continue
        m = {}
        for blk in re.findall(rb"beginbfchar(.*?)endbfchar", s, re.S):
            for a, b in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
                try:
                    m[int(a, 16)] = bytes.fromhex(b.decode()).decode("utf-16-be", "ignore")
                except Exception:
                    pass
        for blk in re.findall(rb"beginbfrange(.*?)endbfrange", s, re.S):
            for a, b, c in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
                lo, hi = int(a, 16), int(b, 16)
                base = int(c, 16)
                for i in range(lo, min(hi, lo + 65535) + 1):
                    try:
                        m[i] = chr(base + (i - lo))
                    except Exception:
                        pass
        if m:
            maps.append(m)
    return maps

def content_text(data):
    chunks = []
    for s in get_streams(data):
        if b"BT" not in s and b"Tj" not in s and b"TJ" not in s:
            continue
        buf = []
        for m in re.finditer(rb"\((?:\\.|[^\\()])*\)|<[0-9A-Fa-f\s]+>|TJ|Tj|T\*|Td|TD|ET", s):
            tok = m.group(0)
            if tok.startswith(b"("):
                body = tok[1:-1]
                body = re.sub(rb"\\([nrtbf()\\])", lambda x: {b"n": b"\n", b"r": b"\r", b"t": b"\t"}.get(x.group(1), x.group(1)), body)
                buf.append(body.decode("latin-1"))
            elif tok.startswith(b"<"):
                hx = re.sub(rb"\s", b"", tok[1:-1])
                if len(hx) % 2:
                    hx += b"0"
                try:
                    raw = bytes.fromhex(hx.decode())
                except Exception:
                    continue
                # try utf-16-be (2-byte CID)
                if len(raw) % 2 == 0:
                    dec = raw.decode("utf-16-be", "ignore")
                else:
                    dec = raw.decode("latin-1")
                buf.append(dec)
            elif tok in (b"T*", b"Td", b"TD", b"ET"):
                buf.append("\n")
        chunks.append("".join(buf))
    return chunks

def main():
    path = sys.argv[1]
    data = open(path, "rb").read()
    maps = tounicode_maps(data)
    chunks = content_text(data)
    combined = "\n".join(chunks)
    # apply tounicode maps where simple 2-byte CIDs appear
    out = combined
    if maps:
        best = max(maps, key=len)
        def sub(m):
            hx = m.group(1)
            if len(hx) % 2:
                return m.group(0)
            try:
                raw = bytes.fromhex(hx)
            except Exception:
                return m.group(0)
            if len(raw) % 2 == 0:
                vals = [int.from_bytes(raw[i:i+2], "big") for i in range(0, len(raw), 2)]
                if all(v in best for v in vals):
                    return "".join(best[v] for v in vals)
            return m.group(0)
        out = re.sub(r"<([0-9A-Fa-f\s]+)>", sub, combined)
    out = re.sub(r"\n{3,}", "\n\n", out)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w", encoding="utf-8").write(out)
        print("wrote", sys.argv[2], len(out), "chars; cmaps:", [len(m) for m in maps])
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(out)

main()
