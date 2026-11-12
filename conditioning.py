#!/usr/bin/env python3
"""
conditioning.py -- one IP-Adapter loader, for the renderer and for previs.

The weights were already cached (`h94/IP-Adapter` + its CLIP-ViT-H image encoder);
this module is the single place that knows where they are, how strong the
conditioning is, and the diffusers quirk that shapes every call: with the adapter
loaded, a call that passes `ip_adapter_image=None` crashes in the UNet. So a
consumer loads it for a conditioned clip/still and unloads it before an
unconditioned one.

`HOLD_CONDITIONING=0` disables it everywhere (renderer AND previs), which is what
makes an A/B meaningful.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

SCALE = 0.7
REPO = "models--h94--IP-Adapter"


def enabled() -> bool:
    return os.environ.get("HOLD_CONDITIONING", "1") != "0"


def snapshot() -> Optional[Path]:
    root = (Path.home() / ".cache" / "huggingface" / "hub" / REPO / "snapshots")
    snaps = sorted(root.glob("*")) if root.exists() else []
    return snaps[-1] if snaps else None


def load(pipe, scale: float = SCALE) -> bool:
    """Attach IP-Adapter to a loaded diffusion pipe. False if unavailable/off."""
    if not enabled():
        return False
    snap = snapshot()
    if snap is None:
        return False
    pipe.load_ip_adapter(str(snap), subfolder="models",
                         weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(scale)
    return True


def unload(pipe) -> None:
    try:
        pipe.unload_ip_adapter()
    except Exception:                              # noqa: BLE001
        pass
