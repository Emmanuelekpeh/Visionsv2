"""The style/palette pass: fog out, one light + palette per scene.

Measured motivation: the 13-shot held films passed coverage but came back as one
murky grey, because every scene shared the same fog/mist words and every palette
was 'neutral'. These tests pin the two properties that made the pass work on the
real store: fog is text-level (so it can be stripped), and the pass is idempotent
(re-running does not pile up light clauses).
"""

from __future__ import annotations

from style_pass import clean, restyle_prompt, scene_style_map


def test_the_plan_palettes_are_exactly_the_grade_palettes():
    """A plan may only name a palette `make_film` can actually apply."""
    import looks
    import story
    assert story.SCENE_PALETTES == looks.ARC


def test_a_look_can_move_a_grey_image_into_green_and_magenta():
    """The measured ceiling of the first pass: a 1-2 Lab tint nudged the muted
    checkpoint palette inside one warm/cool band. The wider looks must actually
    leave that band."""
    import numpy as np
    from PIL import Image
    import cv2
    import looks

    grey = Image.new("RGB", (32, 32), (128, 128, 128))
    hues = {}
    for name in ("emerald", "magenta", "sodium", "teal", "violet"):
        out = looks.apply_look([grey], name)[0]
        lab = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2LAB).astype(np.float32)
        a = lab[..., 1] - 128.0
        b = lab[..., 2] - 128.0
        chroma = float(np.hypot(a, b).mean())
        hue = float(np.degrees(np.arctan2(b.mean(), a.mean())) % 360)
        assert chroma > 5.0, (name, chroma)
        hues[name] = hue
    # green sits on the opposite side of the Lab a axis from magenta
    def side(h):
        return "green" if 90.0 <= h <= 270.0 else "red"
    assert side(hues["emerald"]) == "green"
    assert side(hues["magenta"]) == "red"
    # a hue rotation moves an already-coloured image
    warm = Image.new("RGB", (32, 32), (200, 120, 40))
    rot = looks.Look("rot", chroma=1.0, hue=120.0)
    out = looks.apply_look([warm], rot)[0]
    before = np.asarray(looks.apply_look([warm], "neutral")[0])
    delta = np.abs(np.asarray(out, dtype=int) - before.astype(int)).mean()
    assert delta > 3.0


def test_fog_words_are_stripped_and_text_tidied():
    out = clean("a man on a bench, morning mist shrouding the park, hazy light")
    assert "mist" not in out.lower()
    assert "hazy" not in out.lower()
    assert "shrouding" not in out.lower()
    assert "  " not in out
    assert "a man on a bench" in out


def test_the_light_goes_after_the_anchors_and_before_the_action():
    style = {"light": "hard silver light", "palette": "cold-steel"}
    out = restyle_prompt("a woman, a footbridge, mist, her hand on the rail",
                         style)
    parts = [p.strip() for p in out.split(",")]
    assert parts[0] == "a woman" and parts[1] == "a footbridge"
    assert parts[2] == "hard silver light"
    assert "mist" not in out.lower()


def test_the_pass_is_idempotent():
    style = {"light": "cold blue moonlight", "palette": "cyan-night"}
    once = restyle_prompt("a man, a park, fog, sitting on a bench", style,
                          [style["light"]])
    twice = restyle_prompt(once, style, [style["light"]])
    assert twice == once


def test_each_scene_gets_its_own_palette_in_order():
    styles = [{"light": "a", "palette": "warm-gold"},
              {"light": "b", "palette": "cold-steel"},
              {"light": "c", "palette": "amber-dusk"}]
    m = scene_style_map(["sc1", "sc2", "sc3", "sc4"], styles)
    assert m["sc1"]["palette"] == "warm-gold"
    assert m["sc2"]["palette"] == "cold-steel"
    assert m["sc3"]["palette"] == "amber-dusk"
    assert m["sc4"]["palette"] == "warm-gold"       # cycles, never repeats in a row
    assert len({m[k]["palette"] for k in ("sc1", "sc2", "sc3")}) == 3


def test_keep_fog_actually_keeps_the_fog():
    """`--keep-fog` passed the same text either way and restyle_prompt always
    stripped fog, so the flag did nothing."""
    from style_pass import DEFAULT_STYLES, restyle_prompt

    style = DEFAULT_STYLES[0]
    src = "a bridge, a woman, thick fog rolling in, a lamp"
    assert "fog" not in restyle_prompt(src, style, remove_fog=True)
    assert "fog" in restyle_prompt(src, style, remove_fog=False)
    assert style["light"] in restyle_prompt(src, style, remove_fog=False)
