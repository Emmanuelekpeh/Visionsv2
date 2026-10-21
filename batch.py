import sys, os, re, json, traceback
import pdfgrab

URLS = [
 ("karpenko", "https://graphics.stanford.edu/papers/karpenko_gyro.pdf"),
 ("spark_thesis", "http://stacks.stanford.edu/file/druid:bw930hk0750/thesis_shpark-augmented.pdf"),
 ("golik_ie", "https://image-engineering.de/library/diploma-thesis/696-test-for-image-stabilizers"),
 ("core_pitchrms", "https://core.ac.uk/download/190328136.pdf"),
 ("grossman_leigh", "https://link.springer.com/article/10.1007/BF00247595"),
]

os.makedirs("txt", exist_ok=True)
for name, url in URLS:
    try:
        p = pdfgrab.grab(url, os.path.join("pdfs", name + ".pdf"))
        t = pdfgrab.text(p)
        with open(os.path.join("txt", name + ".txt"), "w", encoding="utf-8") as f:
            f.write(t)
        print(f"OK {name}: {len(t)} chars -> txt/{name}.txt")
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__} {e}")
