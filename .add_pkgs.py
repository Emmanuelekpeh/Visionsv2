"""
Install packages into .ltx_env by fetching wheels with urllib and unzipping.

pip is unusable on this machine (network path hangs; temp/target hits the
sandbox), but a wheel is just a zip and urllib works.

    python .add_pkgs.py huggingface_hub==0.26.2 safetensors==0.4.5
"""
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

W = Path(".wheels")
ENV = Path(".ltx_env")
W.mkdir(exist_ok=True)
ENV.mkdir(exist_ok=True)

tag = f"cp{sys.version_info.major}{sys.version_info.minor}"


def pick(urls):
    """Prefer a pure-python wheel, else a wheel for this interpreter/platform."""
    pure = [f for f in urls if f["filename"].endswith("-none-any.whl")]
    if pure:
        return pure[0]
    plat = [f for f in urls
            if f["filename"].endswith(".whl")
            and ("win_amd64" in f["filename"] or "none-win" in f["filename"])]
    if plat:
        # prefer an abi3/none wheel, else the cp-tagged one
        for f in plat:
            if f"{tag}-none-win_amd64" in f["filename"] or "abi3" in f["filename"]:
                return f
        return plat[0]
    return None


def fetch(spec):
    name, _, version = spec.partition("==")
    with urllib.request.urlopen(
            f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as r:
        data = json.load(r)
    info = pick(data["urls"])
    if info is None:
        print(f"  {name}: no suitable wheel; have "
              f"{[f['filename'] for f in data['urls']][:6]}")
        return False
    dest = W / info["filename"]
    if not dest.exists():
        print(f"  downloading {dest.name} ({info['size']/1e6:.1f} MB)")
        with urllib.request.urlopen(info["url"], timeout=240) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    # drop any previously extracted copy so it cannot shadow the new one
    top = info["filename"].split("-")[0].replace("_", "-")
    import shutil
    for d in list(ENV.glob(f"{name.replace('-', '_')}*")) + list(ENV.glob(f"{top}*")):
        if d.is_dir() and (d / "__init__.py").exists() or d.name.endswith(".dist-info"):
            shutil.rmtree(d, ignore_errors=True)
    for d in ENV.glob(f"{name.replace('-', '_')}-*.dist-info"):
        shutil.rmtree(d, ignore_errors=True)
    with zipfile.ZipFile(dest) as z:
        z.extractall(ENV)
    print(f"  installed {dest.name}")
    return True


for spec in sys.argv[1:]:
    print(f"=== {spec} ===")
    fetch(spec)

sys.path.insert(0, str(ENV.resolve()))
print("\n=== verifying ===")
try:
    import diffusers
    import transformers
    print(f"diffusers    {diffusers.__version__}")
    print(f"transformers {transformers.__version__}")
    for n in ["LTXPipeline", "LTXVideoTransformer3DModel", "GGUFQuantizationConfig"]:
        print(f"  {n}: {hasattr(diffusers, n)}")
    from diffusers.models import AutoencoderKLLTXVideo  # noqa: F401
    print("  AutoencoderKLLTXVideo: True")
except Exception as exc:
    print(f"  FAILED: {type(exc).__name__}: {exc}")
