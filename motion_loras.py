"""
motion_loras.py -- attach motion LoRAs to an AnimateDiff UNet, and prove it.

Two LoRA families are in play:

  camera LoRAs   models/motion_lora/<name>/        official diffusers layout
                 keys: unet.down_blocks.0.motion_modules.0.transformer_blocks.0
                       .attn1.processor.to_q_lora.down.weight

  subject LoRAs  models/subject_lora/<name>.safetensors     Cseti/AD_Motion_LORAs
                 keys: down_blocks.0.motion_modules.0.temporal_transformer
                       .transformer_blocks.0.attention_blocks.0.processor
                       .to_q_lora.down.weight

The subject files differ from diffusers' expected naming in exactly three ways:
`temporal_transformer.` is present, the attention blocks are called
`attention_blocks.0/1` rather than `attn1/attn2`, and there is no `unet.` prefix.
`convert_keys` rewrites them so the proven diffusers loader can take them.

They also carry LoRAs on the feed-forward path (`ff.net.0.proj`, `ff.net.2`).
diffusers' AnimateDiff LoRA loader has nowhere to put those -- it only knows how
to fill `AttnProcessor.<x>_lora` slots -- so they are attached here by hand, as a
delta on the Linear module. That is 64 of the 400 tensors, ~20% of the LoRA, and
silently dropping it would make a working LoRA look weak. Hence the wrappers.

`count_unet_lora_params` exists because of the handoff's standing rule: *count the
tensors before blaming the model*. "It had no effect" is not a verdict until the
weights are proven to be loaded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
CAMERA_DIR = ROOT / "models" / "motion_lora"
SUBJECT_DIR = ROOT / "models" / "subject_lora"
CONVERTED = SUBJECT_DIR / "_diffusers"
FF_DIR = SUBJECT_DIR / "_ff"

# official camera vocabulary vs the subject-motion vocabulary.
# "none" is a first-class option, not an absence: the handoff's directing rule is
# that AnimateDiff already drifts, so forcing a move on every shot is exhausting.
CAMERA_LORAS = ["zoom-in", "zoom-out", "pan-left", "pan-right",
                "tilt-up", "tilt-down", "rolling-clockwise", "rolling-anticlockwise"]
SUBJECT_LORAS = ["walking-frontview", "running-frontview", "ballerina-spin",
                 "casualmove-frontview", "walk-backview", "drone-forward"]


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #

def convert_keys(sd: Dict) -> Tuple[Dict, Dict, Dict]:
    """Cseti/guoyww naming -> diffusers. Returns (attention, feedforward, other)."""
    attn: Dict = {}
    ff: Dict = {}
    other: Dict = {}
    for k, v in sd.items():
        k2 = k[5:] if k.startswith("unet.") else k
        k2 = k2.replace("temporal_transformer.", "")
        k2 = k2.replace("attention_blocks.0.", "attn1.")
        k2 = k2.replace("attention_blocks.1.", "attn2.")
        if "_lora." not in k2:
            other[k] = v
        elif k2.startswith(("down_blocks", "up_blocks", "mid_block")) and \
                ".processor." in k2 and not re.search(r"\.(attn1|attn2)\.processor\.", k2):
            # e.g. down_blocks.0...ff.net.0.processor.proj_lora.down.weight
            ff[k2] = v
        else:
            attn["unet." + k2] = v
    return attn, ff, other


def prepare(name: str, raw: Path | None = None, verbose: bool = True) -> Path:
    """Convert one raw subject LoRA into a diffusers directory. Cached.

    The feed-forward half is written to a SEPARATE directory on purpose: if it
    sat next to the attention weights, diffusers would find two `.safetensors`
    files in the adapter folder and refuse to load either ("Provided path
    contains more than one weights file"). That failure is silent-ish and would
    have made a working LoRA look like it did nothing.
    """
    from safetensors.torch import load_file, save_file

    raw = raw or (SUBJECT_DIR / f"{name}.safetensors")
    out = CONVERTED / name
    ff_path = FF_DIR / f"{name}.safetensors"
    weights = out / "diffusion_pytorch_model.safetensors"
    meta_f = out / "conversion.json"
    if weights.exists() and meta_f.exists() and ff_path.exists():
        return out

    sd = load_file(str(raw))
    attn, ff, other = convert_keys(sd)
    out.mkdir(parents=True, exist_ok=True)
    FF_DIR.mkdir(parents=True, exist_ok=True)
    # clear any stale second safetensors left in the adapter directory
    for stale in out.glob("*.safetensors"):
        if stale.name != weights.name:
            stale.unlink()
    if ff:
        save_file(ff, str(ff_path))
    if other:
        save_file(other, str(SUBJECT_DIR / f"{name}_unmapped.safetensors"))
    save_file(attn, str(weights))
    meta_f.write_text(json.dumps({
        "source": raw.name,
        "total_tensors": len(sd),
        "attention_tensors": len(attn),
        "feedforward_tensors": len(ff),
        "unmapped_tensors": len(other),
        "unmapped_examples": list(other)[:5],
    }, indent=2), encoding="utf-8")
    if verbose:
        print(f"  converted {name}: {len(attn)} attention + {len(ff)} feed-forward "
              f"+ {len(other)} unmapped = {len(sd)}")
    return out


# --------------------------------------------------------------------------- #
# feed-forward LoRAs, attached by hand
# --------------------------------------------------------------------------- #

class _FFLoRA:
    """Adds sum_i w_i * (x @ down_i^T @ up_i^T) to a Linear module's output."""

    def __init__(self, module):
        self.module = module
        self.down: Dict[str, object] = {}
        self.up: Dict[str, object] = {}
        self.weight: Dict[str, float] = {}
        self._orig_forward = module.forward
        module.forward = self._forward

    def add(self, name: str, down, up, weight: float = 1.0):
        import torch
        self.down[name] = torch.nn.Parameter(down, requires_grad=False)
        self.up[name] = torch.nn.Parameter(up, requires_grad=False)
        self.weight[name] = weight
        # register so the tensors move with .to() and appear in state_dict
        self.module.register_parameter(f"_fflora_{name}_down".replace("-", "_"),
                                       self.down[name])
        self.module.register_parameter(f"_fflora_{name}_up".replace("-", "_"),
                                       self.up[name])

    def _forward(self, x, *a, **kw):
        import torch.nn.functional as F
        y = self._orig_forward(x, *a, **kw)
        for name, w in self.weight.items():
            if w == 0.0:
                continue
            d = self.down[name].to(dtype=x.dtype)
            u = self.up[name].to(dtype=y.dtype)
            y = y + w * F.linear(F.linear(x, d), u)
        return y


def _module_for(base: str) -> str:
    """Map a LoRA key's module part onto the module that actually exists.

    The training code wrapped the feed-forward projection in an extra object, so
    the key reads `...ff.net.0.processor.proj_lora` where the real module is
    `...ff.net.0.proj` (the Linear inside the GEGLU). Attaching the delta one
    level too high wraps the GEGLU itself, whose output is half the size, and
    raises "size of tensor a (1280) must match the size of tensor b (2560)".
    """
    if ".processor." in base:
        head, _, tail = base.rpartition(".processor.")
        if tail.endswith("_lora"):
            tail = tail[: -len("_lora")]
        return f"{head}.{tail}"
    mod_path, _, _ = base.rpartition(".")
    return mod_path


class FFBank:
    """All hand-attached feed-forward LoRA deltas, with a per-adapter weight."""
    def __init__(self):
        self.mods: Dict[str, _FFLoRA] = {}
        self.installed: Dict[str, List[str]] = {}

    def install(self, unet, name: str, ff_sd: Dict) -> Tuple[int, List[str]]:
        groups: Dict[str, Dict[str, object]] = {}
        for k, v in ff_sd.items():
            parts = k.rsplit(".", 2)          # [..., 'down', 'weight']
            if len(parts) != 3 or parts[1] not in ("down", "up"):
                continue
            groups.setdefault(_module_for(parts[0]), {})[parts[1]] = v

        attached, missing = [], []
        for mod_path, d in groups.items():
            if "down" not in d or "up" not in d:
                continue
            try:
                mod = unet.get_submodule(mod_path)
            except AttributeError:
                missing.append(mod_path)
                continue
            if mod_path not in self.mods:
                self.mods[mod_path] = _FFLoRA(mod)
            self.mods[mod_path].add(name, d["down"], d["up"], weight=0.0)
            attached.append(mod_path)
        self.installed[name] = attached
        return len(attached), missing

    def set_active(self, name: str | None, weight: float = 1.0):
        for m in self.mods.values():
            for n in m.weight:
                m.weight[n] = weight if n == name else 0.0

    def num_tensors(self) -> int:
        return sum(len(m.down) * 2 for m in self.mods.values())


# --------------------------------------------------------------------------- #
# counting -- the standing rule
# --------------------------------------------------------------------------- #

def count_unet_lora_params(unet) -> int:
    """Number of LoRA parameters currently living in the UNet."""
    return int(sum(p.numel() for n, p in unet.named_parameters() if "lora" in n.lower()))


def count_attached_adapters(pipe) -> List[str]:
    return list(getattr(pipe, "get_active_adapters", lambda: [])())


# --------------------------------------------------------------------------- #
# preflight: prove the LoRA is in the graph before generating an hour with it
# --------------------------------------------------------------------------- #

def verify_effect(pipe, bank: "FFBank", name: str, frames: int = 16, hw: int = 16,
                  seed: int = 0, cross_dim: int = 768) -> float:
    """Max abs change in the UNet output caused by this adapter.

    Four false verdicts in the handoff session traced back to a misconfigured
    run -- including a half-loaded checkpoint. A LoRA that silently failed to
    attach produces a clip that is merely AnimateDiff, and the null result then
    gets blamed on the method. This runs the UNet twice on identical random
    latents, with and without the adapter, and reports the difference.

    0.0 means the weights are not in the graph. That is a harness bug, not a
    result -- so callers should refuse to generate until it is non-zero.
    """
    import torch

    g = torch.Generator("cpu").manual_seed(seed)
    sample = torch.randn(1, 4, frames, hw, hw, generator=g)
    # AnimateDiffPipeline does `prompt_embeds.repeat_interleave(num_frames, dim=0)`
    # before calling the UNet, so the motion modules expect the cross-attention
    # batch to already be B*T. Passing a single (1, 77, dim) embeds raises
    # "size of tensor a (16384) must match the size of tensor b (1024)".
    ehs = torch.randn(frames, 77, cross_dim, generator=g)
    t = torch.tensor([500.0])

    def run():
        with torch.no_grad():
            return pipe.unet(sample, t, encoder_hidden_states=ehs).sample

    try:
        pipe.disable_lora()
        bank.set_active(None)
        a = run()
        pipe.enable_lora()
        pipe.set_adapters([name], adapter_weights=[1.0])
        bank.set_active(name if name in bank.installed else None)
        b = run()
        pipe.disable_lora()
        bank.set_active(None)
        return float((a - b).abs().max())
    except Exception as exc:
        print(f"    verify {name}: could not run the check "
              f"({type(exc).__name__}: {str(exc)[:90]})")
        return -1.0


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #

def catalog() -> Dict[str, dict]:
    """Everything available to the planner, with its kind."""
    out: Dict[str, dict] = {}
    for name in CAMERA_LORAS:
        p = CAMERA_DIR / name
        out[name] = {"kind": "camera", "path": p, "ready": p.exists()}
    for name in SUBJECT_LORAS:
        raw = SUBJECT_DIR / f"{name}.safetensors"
        conv = CONVERTED / name / "diffusion_pytorch_model.safetensors"
        out[name] = {"kind": "subject", "path": raw, "converted": conv,
                     "ff": FF_DIR / f"{name}.safetensors",
                     "ready": raw.exists()}
    out["none"] = {"kind": "none", "path": None, "ready": True}
    return out


if __name__ == "__main__":
    cat = catalog()
    for n, m in cat.items():
        if n == "none":
            continue
        size = ""
        p = m.get("path")
        if p and Path(p).is_dir():
            f = Path(p) / "diffusion_pytorch_model.safetensors"
            size = f"{f.stat().st_size/1e6:.1f} MB" if f.exists() else "?"
        elif p and Path(p).exists():
            size = f"{Path(p).stat().st_size/1e6:.1f} MB"
        print(f"  {n:<24} {m['kind']:<8} {'OK ' if m['ready'] else 'MISSING'} {size}")
    print("\nconverting subject LoRAs...")
    for n in SUBJECT_LORAS:
        if (SUBJECT_DIR / f"{n}.safetensors").exists():
            prepare(n)
