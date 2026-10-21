"""
ui/effects_panel.py -- the effects section.

    from ui.effects_panel import effects_panel
    effects_panel(project)

WHAT THIS IS
------------
A panel for building a CHAIN of image-representation effects on a production, with a
live preview on that production's own frames. The effects themselves live in
`effects.py` (23 of them, five families); this file is only the surface.

Three decisions worth stating, because each one is a way this could have gone wrong:

  * **A CHAIN, not a filter picker.** The payoff is composition -- posterize into
    channel separation into halftone -- so the panel is an ordered list you add to,
    reorder and dial, not a menu where choosing one replaces another.

  * **It previews on REAL frames.** The source is the production's own current takes,
    post-grade, because grade and effects interact: an effect previewed on a raw frame
    can look entirely different after the technical baseline flattens the image. The
    pipeline already has every effect under a second for a whole clip, which is what
    makes a live preview possible at all.

  * **`amount` is first and always visible.** Every effect's zero is exactly the
    original, so a step can be dialled out without being deleted -- which is how you
    find out whether a chain needs a step, rather than guessing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

import effects as fx
import film_store as fs

_V = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
FULL = {"width": "stretch"} if _V >= (1, 49) else {"use_container_width": True}


def rule(label: str = "") -> None:
    """The same quiet section label the studio uses, duplicated here so the panel can
    be imported and rendered on its own without the studio's globals."""
    st.markdown(f'<div class="studio-label">{label}</div>', unsafe_allow_html=True)


def effects_panel(project: Dict) -> None:
    rule("effects")
    steps = fx.chain(project)
    n_shots = len(fs.shots_in_order(project))

    st.caption("Change how the image is represented, printed, degraded or "
               "reconstructed — not a colour filter. Steps run in order, on every "
               "shot's clip, after grading and before the cut. Every step's `amount` "
               "of 0 is exactly the original, so a step can be dialled out without "
               "being removed.")

    # ---- the chain ---------------------------------------------------------- #
    if steps:
        st.markdown(f"**the chain** — {len(steps)} step(s), top to bottom")
    else:
        st.info("No effects. The film renders straight from the grade.")

    for i, s in enumerate(steps):
        name = s.get("effect", "?")
        eff = fx.REGISTRY.get(name)
        amt = float(s.get("amount", 1.0))
        c = st.columns([3, 2, 1, 1, 1])
        label = f"{name}" + ("" if eff else "  (UNKNOWN)")
        c[0].markdown(f"**{i + 1}. {label}**  \n<span style='opacity:.55;font-size:.86em'>"
                      f"{eff.family if eff else '—'} · "
                      f"{eff.blurb if eff else 'not in the registry'}</span>",
                      unsafe_allow_html=True)
        new_amt = c[1].slider(f"amount {name}", 0.0, 1.0, amt, 0.05,
                              key=f"fx_amt_{i}", label_visibility="collapsed")
        if c[2].button("↑", key=f"fx_up_{i}", disabled=(i == 0), **FULL):
            steps[i - 1], steps[i] = steps[i], steps[i - 1]
            fx.set_chain(project, steps)
            fs.save(project)
            st.rerun()
        if c[3].button("↓", key=f"fx_dn_{i}", disabled=(i == len(steps) - 1), **FULL):
            steps[i + 1], steps[i] = steps[i], steps[i + 1]
            fx.set_chain(project, steps)
            fs.save(project)
            st.rerun()
        if c[4].button("✕", key=f"fx_rm_{i}", **FULL):
            steps.pop(i)
            fx.set_chain(project, steps)
            fs.save(project)
            st.cache_data.clear()
            st.rerun()
        if abs(new_amt - amt) > 1e-9:
            steps[i]["amount"] = round(float(new_amt), 3)
            fx.set_chain(project, steps)
            fs.save(project)

        # the effect's own knobs, in a popover so the chain stays readable
        if eff and eff.params:
            with st.popover(f"settings for step {i + 1}", use_container_width=False):
                changed = False
                for k, default in eff.params.items():
                    cur = s.get(k, default)
                    key = f"fx_{i}_{k}"
                    if isinstance(default, bool):
                        v = st.checkbox(k, value=bool(cur), key=key)
                    elif isinstance(default, int):
                        v = st.number_input(k, value=int(cur), step=1, key=key)
                    elif isinstance(default, float):
                        v = st.number_input(k, value=float(cur), step=0.1,
                                            format="%.3f", key=key)
                    elif isinstance(default, str):
                        opts = (["concrete", "sodium", "deepwater", "washed"]
                                if k == "inks" and name == "palette_remap"
                                else [default])
                        v = st.selectbox(k, opts, index=opts.index(cur)
                                         if cur in opts else 0, key=key)
                    else:
                        v = st.text_input(k, value=str(cur), key=key)
                    if v != cur:
                        steps[i][k] = v
                        changed = True
                if changed:
                    fx.set_chain(project, steps)
                    fs.save(project)
                    st.cache_data.clear()
                    st.rerun()

    # ---- add ---------------------------------------------------------------- #
    fams = fx.by_family()
    st.markdown("**add a step**")
    c = st.columns([2, 2, 1])
    fam = c[0].selectbox("family", [f for f in fx.FAMILY_ORDER if fams.get(f)],
                         key="fx_fam")
    choices = [e.name for e in fams.get(fam, [])]
    if not choices:
        return
    pick = c[1].selectbox("effect", choices, key="fx_pick",
                          format_func=lambda n: n.replace("_", " "))
    st.caption(f"*{fams[fam][[e.name for e in fams[fam]].index(pick)].blurb}*")
    if c[2].button("add", key="fx_add", **FULL):
        steps.append({"effect": pick, "amount": 1.0})
        fx.set_chain(project, steps)
        fs.save(project)
        st.cache_data.clear()
        st.rerun()

    if steps:
        b = st.columns([1, 1, 3])
        if b[0].button("clear the chain", key="fx_clear", **FULL):
            fx.set_chain(project, [])
            fs.save(project)
            st.cache_data.clear()
            st.rerun()
        for name in _PRESETS:
            if b[1].button(f"try “{name}”", key=f"fx_pre_{name}", **FULL,
                           help=_PRESETS[name][1]):
                fx.set_chain(project, [dict(s) for s in _PRESETS[name][0]])
                fs.save(project)
                st.cache_data.clear()
                st.rerun()

    # ---- preview ------------------------------------------------------------ #
    if not steps:
        return
    st.divider()
    _preview(project, steps, n_shots)


def _preview(project: Dict, steps: List[Dict], n_shots: int) -> None:
    """Run the chain on a real take's frames and show before / after.

    The source is that shot's delivered frames AFTER the technical baseline and palette,
    which is what the renderer will hand the chain -- so the preview is the export, not
    a different experiment.
    """
    st.markdown("**preview**")
    rows = fs.shots_in_order(project)
    if not rows:
        st.caption("no shots to preview on yet")
        return
    if n_shots > 1:
        c = st.columns([1, 4])
        i = int(c[0].number_input("shot", 1, n_shots, 1, 1, key="fx_prev_shot")) - 1
    else:
        i = 0
    shot = rows[i]
    take = fs.best_take(project, shot)
    if take is None:
        st.caption(f"shot {i + 1} has no take yet, so there is nothing to preview. "
                   f"Shoot it first, then come back.")
        return

    before, after = _preview_frames(project, shot, i, take["_dir"], tuple(
        (s.get("effect"), round(float(s.get("amount", 1.0)), 3),
         tuple(sorted((k, str(v)) for k, v in s.items()
                      if k not in ("effect", "amount"))))
        for s in steps))

    if not before or not after:
        st.caption("could not read this shot's frames, so there is nothing to preview")
        return
    # BEFORE and AFTER as two labelled rows rather than one wide strip: with four steps
    # and three frames each a single row is eight columns and reads as a contact sheet
    # with no before/after boundary. `st.columns` also refuses an empty list, which an
    # earlier version of this handed it when a take had no frames.
    n = min(len(before), len(after), 4)
    st.markdown("**before** — after grading, before effects")
    cols = st.columns(n, gap="small")
    for k in range(n):
        cols[k].image(before[k])
    st.markdown("**after** — the chain applied to those same frames")
    cols = st.columns(n, gap="small")
    for k in range(n):
        cols[k].image(after[k])
    st.caption(f"shot {i + 1} · take {take['take']} · {len(steps)} step(s). This is "
               f"what the export will do — the chain runs on the delivered frames, "
               f"after the grade, per shot.")


def _preview_frames(project: Dict, shot: Dict, index: int, take_dir: str, sig: tuple):
    """(before, after) frame lists, cached on the chain signature.

    The `sig` argument IS the cache key: it encodes the effect order, every amount and
    every named parameter, so changing any knob re-runs the preview and nothing else
    does.
    """
    return _preview_cached(project["slug"], index, take_dir, sig)


@st.cache_data(show_spinner="running the chain on this shot's frames")
def _preview_cached(slug: str, index: int, take_dir: str, sig: tuple):
    # THE IMPORTS ARE INSIDE THE FUNCTION ON PURPOSE. `@st.cache_data` re-executes the
    # function body in a context where module-level names are not reliably resolvable
    # (the decorator wraps and may re-run the source), so `Path` available at module
    # scope is NOT available here. Removing this import to tidy up broke the panel with
    # "name 'Path' is not defined" -- keep it local.
    import looks
    import grade_match as gm
    from pathlib import Path
    from PIL import Image

    project = fs.load(slug)
    shot = fs.shots_in_order(project)[index]
    # `take_dir` arrives as a STRING (`take["_dir"]` is `str(t)` in film_store), so it
    # has to be wrapped before any path operation. `Path(take_dir)` was already correct
    # here -- the earlier "0 frames" readings came from my own probe code doing
    # `str / "frames"`, which is a TypeError. Kept explicit and asserted, because this
    # function is called through a Streamlit cache decorator and an exception inside it
    # surfaces as an empty preview rather than as an error.
    root = Path(take_dir)
    # FRAMES LIVE ONE LEVEL DOWN: `<take_dir>/frames/f000.png`. `make_film` writes them
    # there and reads them back with `Path(take["_dir"]) / "frames"`. Globbing the take
    # directory itself finds only `take.json`, which is why this returned an empty
    # preview with no error -- the `if not src` guard swallowed it.
    frame_dir = root / "frames"
    files = sorted(frame_dir.glob("f*.png")) if frame_dir.is_dir() else []
    src = []
    for p in files:
        try:
            src.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    if not src:
        st.warning(f"no frames found for this take at `{frame_dir}`, so there is "
                   f"nothing to preview.")
        return [], []

    base = src[:max(1, len(src))]
    # grade the way the renderer does, so the effect sees the same input
    try:
        if (project.get("settings") or {}).get("technical_baseline", True):
            base = gm.match_technical(base, gm.technical_target(
                [gm.signature(base)]))
        pal = fs.palette_for(project, shot)
        if pal and pal != "neutral":
            base = looks.apply_look(base, pal)
    except Exception as exc:
        # DO NOT SWALLOW SILENTLY. The first version of this caught everything and
        # passed, so a broken grade produced an empty preview with no explanation --
        # and an exception inside a `@st.cache_data` function surfaces to the user as
        # nothing at all, which is the worst possible failure for a preview.
        st.warning(f"the preview could not reproduce the grade "
                   f"({type(exc).__name__}: {exc}); showing the effect on ungraded "
                   f"frames, which may differ from the export.")

    steps = fx.chain(project)
    after = fx.apply_chain(base, steps, verbose=False)

    def sample(frames, n=3):
        if not frames:
            return []
        if len(frames) <= n:
            return list(frames)
        return [frames[round(k * (len(frames) - 1) / (n - 1))] for k in range(n)]

    size = 200
    return ([f.resize((size, size), Image.LANCZOS) for f in sample(base)],
            [f.resize((size, size), Image.LANCZOS) for f in sample(after)])


# --------------------------------------------------------------------------- #
# presets: compositions, which is where the interesting results are
# --------------------------------------------------------------------------- #

_PRESETS: Dict[str, tuple] = {
    # the user's own examples, near enough to be recognisable
    "print": ([{"effect": "posterize", "amount": 1.0, "levels": 4},
               {"effect": "channel_separation", "amount": 0.7, "offset_px": 4.0},
               {"effect": "halftone", "amount": 0.8, "cell": 5, "irregular": 0.4}],
              "posterize -> channel separation -> halftone"),
    "riso": ([{"effect": "posterize", "amount": 0.9, "levels": 5},
              {"effect": "riso", "amount": 0.85, "inks": 3, "offset_px": 5.0,
               "grain": 0.14}],
             "posterize -> risograph inks"),
    "decay": ([{"effect": "trails", "amount": 0.7, "decay": 0.6},
               {"effect": "film_damage", "amount": 0.6, "dust": 0.5, "scratches": 0.6,
                "weave": 1.4, "flicker": 0.12},
               {"effect": "aberration", "amount": 0.5, "offset_px": 3.0}],
              "trails -> film damage -> aberration"),
    "drawing": ([{"effect": "edges", "amount": 0.9, "keep": 0.3},
                 {"effect": "palette_remap", "amount": 0.6, "inks": "concrete"}],
                "edge extraction -> designed palette"),
    "dissolve": ([{"effect": "displace", "amount": 0.6, "scale": 22.0, "drift": 1.4},
                  {"effect": "painterly", "amount": 0.5, "radius": 5, "strokes": 260},
                  {"effect": "feedback", "amount": 0.4, "gain": 0.3, "zoom": 1.02}],
                 "displacement -> painterly -> feedback"),
    "signal": ([{"effect": "slice", "amount": 0.7, "slices": 14, "shift_px": 26.0},
                {"effect": "pixel_sort", "amount": 0.6, "threshold": 0.62},
                {"effect": "dither", "amount": 0.4, "levels": 3, "ordered": True}],
               "slice -> pixel sort -> dither"),
}
