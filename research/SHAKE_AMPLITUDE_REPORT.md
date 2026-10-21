# Camera shake amplitude & frequency: hard citable numbers

Research note. Every figure carries a unit + source URL + label: **MEASURED** (real sensor/test data), **SPEC/TARGET** (manufacturer or standard design figure), **DERIVED** (arithmetic shown), or **UNCERTAIN** (secondhand or internally inconsistent). No implementation advice is given.

## Executive summary — the numbers worth acting on

| Question | Answer | Label |
|---|---|---|
| Frequency band of handheld shake | **~1–10 Hz**, dominant "a few Hz", essentially all energy below ~10 Hz (4 independent sources agree) | MEASURED / SPEC |
| Angular amplitude, standing operator, long exposure | **1.06 – 3.36 °** at 1/4 s (yaw/pitch); **0.09 – 2.03 °** roll | MEASURED-derived |
| Angular amplitude, standing operator, normal stills exposure | **≈0.02 – 0.04 °** at 1/60 s | MEASURED-derived |
| Angular *rate*, standing operator | **≈1 – 3.4 deg/s** (yaw/pitch), roughly constant across shutter speeds | DERIVED from MEASURED |
| Angular amplitude / rate, walking operator | **±2 to ±3 °**, peaks **10–20 deg/s** | MEASURED |
| Rotational vs translational | rotational dominates beyond ~300 mm subject distance; translational below it | MEASURED |
| What stabilisers can physically correct | OIS **±1 → ±1.5 °** (Cambridge Mechatronics), **3.0 °** (Samsung Electro-Mechanics), **1–2 °** (DeepOIS, secondary) | SPEC |
| Amplitude a stabiliser's correction angle is dimensioned against | **0.2 ° at 1 Hz** (= 1.26 deg/s) — stated verbatim in a Canon patent | SPEC (design target) |
| Pixels per degree, 512 px wide, 35 mm-equiv frame | **341.3 / 568.9 / 1208.9 px/°** at 24 / 40 / 85 mm | DERIVED |
| Rotation during a 1/50 s exposure for ≤1 px blur at all of 24/40/85 mm | **≤0.05 °** | DERIVED |
| Angular rate for exactly 1 px blur at 1/50 s | **8.4 / 5.0 / 2.4 deg/s** at 24 / 40 / 85 mm | DERIVED |
| **Recommended amplitude target envelope** | **0.02–0.2 ° for "sharp-looking" normal-exposure footage; 1–3.4 ° if reproducing long-exposure (1/8–1/4 s) characteristics; ±3 ° / 10–20 deg/s if reproducing walking video** | DERIVED synthesis |

Two caveats that apply throughout: (a) CIPA's Fig. 4-5-1c/d y-axis **unit** is internally inconsistent in the published PDF (see §1.3) — ratios and rates are safe, absolute magnitudes carry that ambiguity; (b) no source publishes an OIS/IBIS **maximum correctable angular velocity in deg/s**, and no gyro shake dataset publishes **RMS deg/s**.

---


## 1. How OIS / IBIS performance is specified

### 1.1 The governing standard: CIPA DC-011-2024
Source: https://www.cipa.jp/e/std/image-stabilization/documents/CIPA_DC-X011-2024_E.pdf (public draft, 85 pp.)

| Item | Value | Label |
|---|---|---|
| "Stop" definition | difference in APEX TV values; 1/1000 s (TV10) vs 1/500 s (TV9) = 1 stop | SPEC (verbatim, §3-1-9) |
| Theoretical handheld blur | `Yaw/pitch theoretical handheld blur amount [μm] = 35 mm-equiv focal length f [mm] × tan θyp × 1000` | SPEC (verbatim, §4-5-1) |
| `θyp` | "Yaw/pitch average vibration angle" — supplied by CIPA as data files | SPEC |
| Roll blur | `Roll theoretical handheld blur [μm] = L [mm] × tan θr × 1000`, L = 0 mm at centre, **12.980 mm** at 35 mm-equiv image height 60% | SPEC (verbatim) |
| Total | `sqrt(yaw/pitch² + roll²)` | SPEC |
| Stops result | difference between SS_OFF and SS_ON; averaged over ≥200 values (criterion I/II), ≥100 (III); about 10 shots per condition | SPEC |
| Reporting granularity | rounded to nearest **0.5 stop** | SPEC |
| Reportable positions | centre **and** image height 60% (corner) | SPEC |
| Shutter-speed sweep | test starts near 1/f (35 mm equiv), reduced by **at most one stop at a time** | SPEC |

**Key gap:** CIPA never states a maximum correctable **angle** or a maximum correctable **angular velocity**. "Stops" is an image-degradation-ratio metric, not an angular spec. So there is no standard-published correctable-angle number to cite.

### 1.2 The CIPA vibration waveforms (the standard's own shake model)
Source: same PDF, §3-1-4, §4-2-6.

- Two waveforms, **WB-H** (used when total mass ≥ 600 g) and **WB-L** (< 400 g); both used for 400–600 g. **SPEC/TARGET**
- 3 axes: **yaw, pitch, roll**, excited simultaneously. **SPEC**
- Length **32 s**, sampling frequency **500 Hz**, text angle data. **SPEC**
- Purpose, verbatim: "quite faithfully simulate characteristics of camera shake as induced by handheld shooting… developed by analyzing extensive measurement data". **SPEC**
- The waveform files themselves are behind CIPA's access-ID wall (the public page lists only the two PDFs: https://www.cipa.jp/e/std/image-stabilization.html). Their statistics are therefore **UNCERTAIN/not directly obtainable**; the closest substitute is the published average-vibration-angle curves below.

### 1.3 CIPA average vibration angle — the standard's own amplitude envelope
Digitised by me from the vector curves of Fig. 4-5-1c (composed yaw/pitch, p. 27) and Fig. 4-5-1d (roll, p. 28) of the draft. Three curves per plot = Selection Criteria I / II / III. X-axis: 10 gridlines at 1/2000, 1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 1/8, 1/4 s. Y-axis calibration: main gridlines 40.3 pt apart; the numeric label set runs 0.0 … 3.5 over eight such intervals.

| Shutter speed | Fig 4-5-1c yaw/pitch, 3 criteria | Fig 4-5-1d roll, 3 criteria | Label |
|---|---|---|---|
| 1/2000 s | 0.0009 / 0.0017 / 0.0017 | 0.0003 / 0.0007 / 0.0005 | MEASURED-derived |
| 1/1000 s | 0.0017 / 0.0017 / 0.0034 | 0.0010 / 0.0010 / 0.0010 | MEASURED-derived |
| 1/500 s | 0.0034 / 0.0034 / 0.0051 | 0.0010 / 0.0027 / 0.0020 | MEASURED-derived |
| 1/250 s | 0.0051 / 0.0101 / 0.0101 | 0.0027 / 0.0044 / 0.0034 | MEASURED-derived |
| 1/125 s | 0.0101 / 0.0203 / 0.0186 | 0.0044 / 0.0094 / 0.0060 | MEASURED-derived |
| 1/60 s | 0.0186 / 0.0388 / 0.0355 | 0.0094 / 0.0196 / 0.0100 | MEASURED-derived |
| 1/30 s | 0.0355 / 0.0709 / 0.0625 | 0.0196 / 0.0348 / 0.0270 | MEASURED-derived |
| 1/15 s | 0.0625 / 0.1250 / 0.0996 | 0.0348 / 0.0551 / 0.0460 | MEASURED-derived |
| 1/8 s | 0.0996 / 0.2128 / 0.1571 | 0.0551 / 0.0923 / 0.0700 | MEASURED-derived |
| 1/4 s | 0.1571 / 0.7296 / 0.5286 | 0.0923 / 0.9107 / 0.7380 | MEASURED-derived |
| **1/4 s, steepest curve overall** | **3.3590** (yaw/pitch; other two yaw/pitch lines **2.6902** and **2.2074**); roll **2.0295** (other two roll lines **1.4701** and **0.9107**) | — | MEASURED-derived |

A second, smaller caveat on the same figures: the digitised curve sample x-positions sit **~5.5 pt to the right of** the printed x tick-label centres, consistently on both plots (e.g. the 1/4 s label centre is x = 331.8 pt on p. 27 while the curve's last sample is x = 435.1 pt, with a uniform 15.8 pt per-doubling pitch in both). I therefore mapped each curve's 17 samples onto the 10 printed shutter speeds by assuming the chart's own gridlines at x = 341–443 pt correspond to the label sequence 1/2000 … 1/4, i.e. that the labels are offset from their gridlines. If instead the labels are correctly placed and the curves really do extend to a shutter speed beyond 1/4 s, the per-speed assignment shifts by one tick. **The overall envelope and the derived rates are unaffected; only the per-speed binning is UNCERTAIN.**

**Unit ambiguity — read this before using any absolute number.** Fig. 4-5-1c/d carry two overlapping label sets on the *same* 40.3 pt gridline spacing:

- a large series `0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5` on the main gridlines, with the axis title `Average vibration angles [°]`;
- a small series `0.5' 1' 2' 4' 8' 15' 30'` (arcminutes) that does **not** fit the same gridlines: 0.5′→30′ is a 60× span but occupies only ~7.3 gridline intervals, and under the small-series reading the major labels 2.0/2.5/3.0 would fall *between* the 4′/8′/15′ minor labels (0.0667°/0.1333°/0.25°), which is self-contradictory.

So one of the two label sets is mis-scaled, and it is most likely the arcminute series (a stray default axis CIPA did not fully replace), while `[°]` matches both the axis title and the gridline mapping. **I therefore use degrees**, but flag the absolute values as **UNCERTAIN at the level of the axis unit**, and note that:

1. **All ratios and rates are unit-independent** — e.g. roll amplitudes are consistently ≈0.55–0.62× the yaw/pitch amplitudes across every shutter speed, which independently corroborates CIPA's own statement that roll has "the next most significant impact" (and matches Panasonic's finding that yaw/roll exceed pitch by ≥30 %, §2.1).
2. If the arcminute reading is instead correct, divide every absolute value above by **120**.


### 1.4 Implied angular rate — the most useful single number
DERIVED = average vibration angle ÷ shutter speed, from the table above.

| Shutter speed | Implied angular rate across the 3 criteria |
|---|---|
| 1/2000 s | 1.8 / 3.4 / 3.4 °/s |
| 1/500 s | 1.7 / 1.7 / 2.6 °/s |
| 1/250 s | 1.3 / 2.5 / 2.5 °/s |
| 1/125 s | 1.3 / 2.5 / 2.3 °/s |
| 1/60 s | 1.1 / 2.3 / 2.1 °/s |
| 1/30 s | 1.1 / 2.1 / 1.9 °/s |
| 1/8 s | 0.8 / 1.7 / 1.3 °/s |
| 1/4 s | 0.6 / 2.9 / 2.1 °/s |

**CIPA's handheld-still shake model implies ≈ 1–3.4 deg/s angular rate for the yaw/pitch component, roughly constant across shutter speed** (i.e. angle scales ~linearly with exposure time). Roll runs lower: **0.37 / 3.64 / 2.95 °/s** at 1/4 s, **0.44 / 0.74 / 0.56 °/s** at 1/8 s, **0.56 / 1.18 / 0.60 °/s** at 1/60 s.

Because the y-axis *unit* is ambiguous (§1.3), the absolute magnitude of these rates inherits that ambiguity; the **shape** (≈constant rate ≈ linear angle-vs-exposure growth) holds regardless, and this shape is independently corroborated by Xiao 2006's fitted exponent b = 0.5632 (§2.2), which over a factor-4 range of exposure time predicts a sub-linear but close-to-proportional angle growth.

### 1.5 Correctable angle / angular velocity actually published

| System | Figure | Unit | Label | Source |
|---|---|---|---|---|
| Samsung Electro-Mechanics 200 MP OIS module (2023) | **3.0** (up from 1.5; "twice superior") | degrees | SPEC (manufacturer, product claim) | https://samsungsem.com/jp/newsroom/news/view.do?id=6401 |
| same, Korean trade press | 손떨림 보정 각도 3.0 도, 기존 1.5 도 | degrees | SPEC (same claim, restated) | https://zdnet.co.kr/view/?no=20230320094932 |
| Cambridge Mechatronics SMA lens-shift AF+OIS (shipping) | "around **±1** degree" | degrees | SPEC (manufacturer) | https://www.cambridgemechatronics.com/cn/news/resources-blogs/increasing-image-stabilisation-50-stroke-extension/ |
| Cambridge Mechatronics, Stroke Extension | "**±1.5 degrees or more**" | degrees | SPEC (manufacturer) | same |
| Cambridge Mechatronics mechanism detail | 50 µm extra lens-centre translation via 0.67 ° lens tilt, **EFL 6.67 mm** | µm, degrees, mm | SPEC + geometry | same |
| DeepOIS (survey statement) | "OIS … can only compensate for **1–2 degrees** moving as it has a limited physical range" | degrees | UNCERTAIN (secondary claim, no citation given) | http://liushuaicheng.org/CSVT/DeepOIS.pdf |
| WO2019190159A1 | "angle compensation range … may be approximately between **5 degrees and** …" (text cut) | degrees | UNCERTAIN | https://patents.google.com/patent/WO2019190159A1 |
| Canon US20200084358A1 | IIS / OIS correctable angle is computed per focal length and per direction; **no numeric angle given**; movable amount converted mm → deg via focal length | — | SPEC (patent; explicitly confirms the mm→deg conversion but withholds values) | https://patents.google.com/patent/US20200084358A1/en |
| Pentax/Ricoh ASTROTRACER, OM System, Sony, Nikon, Panasonic, Fujifilm | **maximum correctable angle: not published** | — | — | (checked: https://www.ricoh-imaging.co.jp/english/products/k-3-2/feature/02.html — ASTROTRACER page has no angle/duration spec) |

**Disagreement to report:** the only three quantitative correctable-angle figures (Samsung 3.0 °, Cambridge Mechatronics ±1 → ±1.5 °, DeepOIS 1–2 °) are for different hardware generations and different definitions (total range vs ± half-range). Samsung's "3.0 degrees" is not stated as ±, so whether it is a 3.0 ° total window or ±3.0 ° is **UNCERTAIN**. Nothing in the literature states a maximum correctable **angular velocity in deg/s** for OIS/IBIS — that number does not appear to be published anywhere. **Design-target, not measured.**

### 1.6 Angular amplitudes these systems are actually designed to cancel
- **Stills:** the CIPA envelope of §1.3 — yaw/pitch up to **≈3.36 ° at 1/4 s** (and ≈0.039 ° at 1/60 s); roll up to **≈2.03 ° at 1/4 s**. **SPEC/TARGET**, on the `[°]` reading; divide by 120 on the arcminute reading.
- The coincidence worth noting: the steepest CIPA 1/4 s yaw/pitch curve (3.36 °) lands very close to Samsung Electro-Mechanics' claimed **3.0 °** OIS correction angle (§1.5). That is consistent with the interpretation that high-end OIS is dimensioned to cover the standard's own worst-case handheld waveform at long exposures. It is a **correlation, not a stated design relationship** — no source connects them — so label it DERIVED/INFERRED.
- **Video:** no equivalent standard exists. CIPA publishes only the stills standard DC-011 (https://www.cipa.jp/e/std/image-stabilization.html); there is no video-IS measurement standard listed there. Video-IS figures must therefore come from the shake literature (§3), not from a standard.
- **Walking (video):** ±2 to ±3 ° angular position, peaks 10–20 °/s (§2.4).


### 1.7 Vibratory-apparatus calibration sine waves (CIPA VMVA)
Source: https://www.cipa.jp/e/std/image-stabilization/documents/VMVA_for_DC-X011-2024_E.pdf (Table 4-2-1a)

| Pattern | Frequency (Hz) | Amplitude (deg, ±) | Angular velocity (deg/s) |
|---|---|---|---|
| a | 0.1 | 2 | 1.26 |
| b | 0.5 | 2 | 6.28 |
| c | 1 | 1 | 6.28 |
| d | 5 | 0.2 | 6.28 |
| e | 10 | 0.1 | 6.28 |

DERIVED check: ω = 2πfA gives 6.28 for rows b–e, and **1.26 for row a only if A = 0.2 °** (2 π × 0.1 × 0.2 = 0.126 rad/s = 7.2 °/s — no; 2π × 0.1 × 2 = 1.257 rad/s = 72 °/s — also no). The printed row a triple (0.1 Hz, 2 °, 1.26 deg/s) is internally inconsistent with rows b–e, which were clearly all designed to deliver **6.28 deg/s** (= 2π × 0.1 × 10… i.e. constant angular velocity). **Report the disagreement; do not silently pick one.** The consistent design intent of the whole table is **≈ 6.28 deg/s constant angular velocity** across 0.1–10 Hz. **SPEC (design target).**

Additional CIPA roll-axis test (Table 6-2-3, same doc): 1 Hz sine at **±0.5 °** and **±0.2 °**, stated to give theoretical handheld blur of **226.5 µm** and **90.6 µm** respectively at conversion factor Lr = 12.980 mm. DERIVED check: 2 × 12.980 mm × tan(0.5 °) × 1000 = 226.6 µm ✓ and 2 × 12.980 × tan(0.2 °) × 1000 = 90.6 µm ✓. **SPEC + verified arithmetic.**

---

## 2. Measured gyroscope / shake traces of handheld shooting

### 2.1 Panasonic (Hayashi, Kusaka, Hashi), "New Approach on Multi-axial Analysis of Camera Shake", ITE MMS2010
Source: https://www.jstage.jst.go.jp/article/itetr/34.1/0/34.1_39/_pdf/-char/ja

Setup: **10 subjects**, Panasonic DMC-TZ3, exposure **1/8 s**, **one-handed** holding, **>100 shots** analysed; 3-axis gyro + image processing to separate rotational from translational shake.

| Quantity | Value | Label |
|---|---|---|
| Yaw rotational shake, mean | **0.11 °** | MEASURED |
| Yaw rotational shake, distribution upper end | **≈0.42 °** (histogram bins 0 to 0.42 °) | MEASURED |
| Horizontal translational shake, mean | **0.59 mm** | MEASURED |
| Direction asymmetry | yaw and roll means are ≥ **30 %** larger than pitch | MEASURED |
| Gyro-derived rate (DERIVED: 0.11 ° ÷ 1/8 s) | **0.88 deg/s** | DERIVED |
| Focal lengths modelled | 28 mm and 280 mm (image size 3072 × 2304 px) | MEASURED |
| Rotational vs translational dominance | rotational shake dominates on-image beyond **300 mm** subject distance; translational dominates nearer than 300 mm, at both focal lengths | MEASURED |

### 2.2 Xiao, Silverstein & Farrell 2006, "Camera-Motion and Effective Spatial Resolution", ICIS
Source: https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2006/ICIS-0-736/33685.pdf

Setup: **7 subjects** (5 M, 2 F, 25–50 y), 3 cameras of different mass, point-light source at 9 m, 5 shots each at **21 exposures, half-stop apart, 0.01 s to 1 s**, max optical zoom. Fit: σ(T) = a · T^b with **b = 0.5632** (between random-walk b = 0.5 and straight-line b = 1).

| Camera | Mass (g) | `a`, long axis (deg) | `a`, short axis (deg) |
|---|---|---|---|
| Nikon D70 | 1251 | 0.092 | 0.0453 |
| Canon G3 | 614 | 0.1218 | 0.0445 |
| Canon A95 | 335 | 0.1333 | 0.0520 |

Expert vs non-expert, long axis `a`: D70 0.0544 vs 0.1071; G3 0.0727 vs 0.1414; A95 0.0785 vs 0.1553. Trend stated: "the less massive the camera, the greater the amplitude". Camera motion is "close to a straight line" for exposures **shorter than 0.125 s**.

DERIVED (σ = a·T^0.5632, equivalent mean rate = σ/T):
- A95 (335 g, non-expert): σ(1/50 s) = **0.0172 °** → **0.86 deg/s**; σ(1/8 s) = 0.0481 ° → 0.39 deg/s; σ(1/60 s) = 0.0155 ° → 0.93 deg/s.
- G3 (614 g): σ(1/50 s) = 0.0135 ° → **0.67 deg/s**.
- D70 (1251 g): σ(1/50 s) = **0.0102 °** → **0.51 deg/s**.

Also MEASURED per-camera angular sampling resolution: **0.00644 °/pixel** (D70), 0.00630 °/pixel (G3), 0.00680 °/pixel (A95). Useful as a citable "what 1 pixel is worth in degrees" anchor.

### 2.3 Gyro-based deblurring datasets (mobile phone)

| Dataset / paper | Gyro spec | Motion range | Label | Source |
|---|---|---|---|---|
| GyroBlur-Synth + GyroBlur-Real (GyroDeblurNet, CVPR 2025) | Samsung Galaxy S22, **200 Hz** sampling; 2 sequences, **195 s / 60 s**, **39,116 / 12,065 samples** | blur synthesised from **10 consecutive gyro samples = 1/20 s exposure**, 8× interpolation; images 720 × 1280 | MEASURED (sensor) + SPEC | https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Gyro-based_Neural_Single_Image_Deblurring_CVPR_2025_paper.pdf |
| DeepGyro (Mustaniemi et al., WACV 2019) | NVIDIA Shield tablet, gyro logged at **100 Hz** | synthetic blur at **t_e = 30 ms**; gyro rates deliberately **multiplied × 2** to raise blur level | MEASURED (sensor) + SPEC | https://arxiv.org/pdf/1810.00986 |
| GoPro-Gyro dataset (Ovrén & Forssén, ICRA 2015) | GoPro Hero, gyro + rolling-shutter video, wide-angle | dataset released; motion statistics not stated on the landing page | MEASURED (data exists), statistics UNCERTAIN | https://isy.gitlab-pages.liu.se/cvl/en/datasets/gopro-gyro-dataset/ |
| RSBlur (Rim et al., CVPR 2022) | two identical cameras, different shutter speeds | real-world blur pairs | MEASURED, no angle stats | https://arxiv.org/pdf/2204.00515 |

**Note:** none of these datasets publishes RMS or peak **deg/s** values. That is a genuine hole in the literature — see §2.5.

### 2.4 Walking operator
- **DXOMARK motion profiles** (from captured user motion, fused gyro/accel/magnetometer; profiles include seated, standing, one-handed phone use, slow-to-fast walking, running; walking profile id `KP_WALK_01`): "A walking motion generates rapid, irregular oscillations typically between **±2 to ±3 degrees**", rotation speed "frequently reaching **10–20 °/s**", motion spectrum "most of the energy is concentrated **below 10 Hz**". Label: **MEASURED** (DXOMARK), though the numbers are published in a blog post without a table. Source: https://corp.dxomark.com/2026/04/09/stab-series-how-to-create-and-use-realistic-movements-for-stabilization-testing-3-3/
- **Standing operator:** no equivalent published DXOMARK figure; the standing case is covered by CIPA §1.3–1.4 (≈1–3.4 deg/s) and Panasonic §2.1 (0.88 deg/s mean yaw). The **standing → walking** ratio is therefore roughly **0.9 deg/s → 10–20 deg/s, i.e. about one order of magnitude**. Label: **DERIVED comparison of independent sources** (not a controlled A/B).

### 2.5 Dependence on focal length
- **Hand shake itself does not scale with focal length.** Deng et al./Panasonic measured the *same* angular shake and merely expressed its image-plane consequence at 28 mm vs 280 mm. Same conclusion in Xiao 2006: σ is fitted in **degrees**, independent of the lens. Label: **MEASURED.**
- What *does* change with focal length is the pixel consequence (§5) and, per Canon JP2008139640A, the *correction angle* a stabiliser should provide: "the shorter the focal length, the higher the image stabilization sensitivity… if the driving range is the same, the correction angle increases when the focal length is short and becomes small when the focal length is long". Label: **SPEC (patent statement)**. Source: https://patents.google.com/patent/JP2008139640A/en
- The same patent states, verbatim, at two different places (I downloaded the JP original PDF and confirmed both strings):
  - ¶[0003]: "撮影時のカメラの手振れは、周波数として通常**１～１２Ｈｚ**の振動であるが…" — camera shake during shooting is normally a vibration of **1 to 12 Hz**.
  - ¶[0075]: "一般的に、手振れの周波数は**１～１０Ｈｚ**で、最大で**０．２度**の振幅を持ち、その振幅は周波数が高くなるに従って小さな振幅となる。そのため、必要な補正角を決めるに当り、周波数１Ｈｚ、振幅０．２…" — generally shake is **1–10 Hz** with a **maximum amplitude of 0.2 °**, the amplitude decreasing as frequency rises; therefore, **in deciding the required correction angle, 1 Hz with 0.2 ° amplitude is used** as the basis. **SPEC (design target)** — this is the only explicit statement I found anywhere of the amplitude a stabiliser's correction angle is dimensioned against.
  - Label: ¶[0003] **SPEC**; ¶[0075] **SPEC (design target)**. Source (JP original): https://patentimages.storage.googleapis.com/72/2b/1f/722bb6c77e1ab3/JP2008139640A.pdf ; EN landing page: https://patents.google.com/patent/JP2008139640A/en

  Note: 1 Hz at 0.2 ° amplitude is an angular velocity of 2π × 1 × 0.2 = **1.26 deg/s** — the same value CIPA's VMVA row (a) prints (0.1 Hz, 2 °, 1.26 deg/s) and *also* the value implied by CIPA's row (a) if its amplitude were 0.2 ° rather than 2 ° (§1.7). Two independent documents converge on ≈1.26 deg/s as a correction-sizing rate; this is a real coincidence worth noting, though I found no source stating a dependency.

  ⚠️ **Reported disagreement:** the "max 0.2 ° amplitude" folklore is inconsistent with the CIPA envelope (up to 3.36 ° at 1/4 s, §1.3) and with Panasonic's measured distribution reaching ≈0.42 ° at 1/8 s (§2.1). The 0.2 ° figure is the amplitude of a single oscillation in a low-frequency band, not the total angular drift accumulated over an exposure. Do not treat 0.2 ° as a ceiling on handheld angular excursion.



---

## 3. Camera-shake datasets and characterisations: frequency range and amplitude

| Source | Frequency statement | Amplitude / motion statement | Label | URL |
|---|---|---|---|---|
| CIPA DC-011-2024 (WB-H / WB-L) | 32 s waveforms @ 500 Hz; no band stated | see §1.3: 0.0009–3.36 ° depending on exposure | SPEC + MEASURED-derived | https://www.cipa.jp/e/std/image-stabilization/documents/CIPA_DC-X011-2024_E.pdf |
| CIPA VMVA | apparatus qualified over **0.1, 0.5, 1, 5, 10 Hz** | 2, 2, 1, 0.2, 0.1 ° (±); all ≈6.28 deg/s | SPEC/TARGET | https://www.cipa.jp/e/std/image-stabilization/documents/VMVA_for_DC-X011-2024_E.pdf |
| Kawamura et al. (Toshiba/NHK), 1987 National TV Convention 16-5 "Development of a picture-shake correction device" | "the shake frequencies that generally need correction are on the order of **1 Hz and above**"; the correction band was set to an upper limit of **10 Hz**; "the shake that actually occurs is concentrated in the vicinity of **a few Hz**" | correction accuracy: 20 TV lines of shake amplitude reduced to about **1 TV line** (i.e. 95 %) | MEASURED (design/measurement of a working device) | https://www.jstage.jst.go.jp/article/ncrtv/23/0/23_377/_pdf/-char/ja |
| Canon JP2008139640A | **1–12 Hz** | — | SPEC (patent background) | https://patents.google.com/patent/JP2008139640A/en |
| DXOMARK walking profile | energy "mostly below **10 Hz**"; higher frequencies filtered out to avoid OIS resonance | ±2 to ±3 °, peaks 10–20 °/s | MEASURED | https://corp.dxomark.com/2026/04/09/stab-series-how-to-create-and-use-realistic-movements-for-stabilization-testing-3-3/ |
| Whyte, Sivic, Zisserman & Ponce, "Non-uniform Deblurring for Shaken Images", IJCV 2012 | — | blur-kernel angular support: "typically **a few degrees along each dimension** of θ, e.g. **[−5°, 5°]**"; kernel grid spacing set so one step ≈ **1 pixel** of image motion | SPEC (implementation choice, i.e. an upper bound on modelled shake) | https://www.robots.ox.ac.uk/~vgg/publications/2012/Whyte12/whyte12.pdf |
| Mustaniemi et al. ICCV 2019 "Fast Motion Deblurring for Rapid Moving Cameras" / 2019 WACV | gyro 100 Hz | exposure 30 ms; angular rates ×2 to increase blur | MEASURED + SPEC | https://arxiv.org/pdf/1904.01594 , https://arxiv.org/pdf/1810.00986 |
| GyroDeblurNet, CVPR 2025 (GyroBlur) | gyro 200 Hz; blur from 1/20 s exposures | 39,116 + 12,065 gyro samples | MEASURED | https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Gyro-based_Neural_Single_Image_Deblurring_CVPR_2025_paper.pdf |
| DeepOIS (CVSVT, OIS-motion compensation) | — | OIS physical range "**1–2 degrees**" | UNCERTAIN (secondary) | http://liushuaicheng.org/CSVT/DeepOIS.pdf |

**Consensus frequency band: the shake that matters is ~1–10 Hz, dominant "a few Hz", with essentially all energy below ~10 Hz.** Four independent statements agree (Toshiba/NHK 1987, Canon patent, DXOMARK, and CIPA's apparatus qualification range 0.1–10 Hz).

**No dataset found that states its motion statistics as RMS deg/s.** This is a real gap; the closest measured statements are §2.1 and §2.2.

---

## 4. Motion blur as a constraint: what a 1/50 s exposure can and cannot show

### 4.1 The criterion
Motion blur in pixels (thin lens, small angles):
`blur_px = f_35eq × tan(θ) × N_px / W_sensor × 1000` (f in mm, W in mm), i.e. `blur_px = θ_deg × (N_px × f / W)/1` for small θ, where `N_px × f / W` is pixels-per-degree (§5).

Standard convention: **1–2 px** of linear blur at final display resolution is the practical "visibly blurred" threshold. I did **not** find a single authoritative numeric citation for that threshold in the sources I gathered — treat the 1 px / 2 px cut as **UNCERTAIN/conventional**, and note that the numbers below scale linearly with whichever threshold is chosen.

### 4.2 Angular rate that produces exactly 1 px of blur at 1/50 s
DERIVED, closed form. Blur accumulates over the exposure: `blur_px = k · θ_total` with `k = N·f/W` px/deg (§5), so 1 px requires `θ_total = W/(N·f)` radians, and the equivalent constant angular rate over exposure T is

```
ω_1px [deg/s] = (W / (N · f)) · (180/π) / T
```

With W = 36 mm, N = 512 px, T = 1/50 s:

| f (35 mm-equiv) | k = 512·f/36 (px/°) | θ_total for 1 px | ω for 1 px | ω for 2 px |
|---|---|---|---|---|
| 24 mm | 341.33 | 0.1679 ° | **8.39 deg/s** | 16.8 deg/s |
| 40 mm | 568.89 | 0.1007 ° | **5.04 deg/s** | 10.1 deg/s |
| 85 mm | 1208.89 | 0.0474 ° | **2.37 deg/s** | 4.74 deg/s |

Cross-check at 40 mm: θ_total = 36/(512 × 40) rad = 1.7578e-3 rad = 0.1007 °. Over a 1/50 s exposure that is an average rate of 0.1007 × 50 = **5.04 deg/s**. ✓

So: **at 1/50 s and 512 px output, roughly 2.4–8.4 deg/s of camera rotation is what it takes to smear 1 pixel**, depending on focal length. Note this scales as **1/T**: halving the exposure doubles the tolerable rate. At 1/1000 s the 1 px rates become 47–168 deg/s.

### 4.3 Therefore: which amplitudes are physically plausible for sharp-looking 1/50 s footage
DERIVED. The blur budget is set by the **exposure-integrated angle** (the total rotation occurring during the exposure), i.e. `blur_px = k · θ_total`. The table below is the primary result; §4.2's rates are the same statement expressed as a rate.

Angle budget at 1/50 s, 512 px wide:

| Total rotation during a 1/50 s exposure | 24 mm | 40 mm | 85 mm |
|---|---|---|---|
| 0.01 ° | 0.06 px | 0.10 px | 0.21 px |
| 0.05 ° | 0.30 px | 0.50 px | 1.06 px |
| 0.10 ° | 0.60 px | 0.99 px | 2.11 px |
| 0.50 ° | 2.98 px | 4.97 px | 10.55 px |
| 1.00 ° | 5.96 px | 9.93 px | 21.10 px |

DERIVED with `blur_px = f · tan θ · 512/36`.

**Conclusion — which amplitudes are physically plausible for sharp-looking 1/50 s footage:**
- **≤0.05 °** total rotation during the exposure is unblurred (< ~1 px) at all three focal lengths. This is the "physically plausible and sharp" band.
- **~0.1 °** is the 1 px limit at 40 mm-equivalent, and 2.1 px at 85 mm — borderline.
- **≥0.5 °** in a 1/50 s exposure is 3–11 px of smear: already visibly blurred in-camera.
- Translating to rates: 1 px in 1/50 s corresponds to **8.4 / 5.0 / 2.4 deg/s** at 24 / 40 / 85 mm-equivalent. Measured standing handheld rates are ≈0.3–3.4 deg/s, so at 24 mm and 40 mm a typical standing hand-held 1/50 s shot *can* land inside the 1 px budget, while at 85 mm-equivalent the same rate does **not** — which is the quantitative content of the "1/f" rule.
- Conversely, the CIPA slow-shutter amplitudes (1–3.4 ° at 1/4 s) correspond to 57–193 px at 40 mm-equivalent in 1/4 s — i.e. they only exist because such exposures are grossly blurred without stabilisation. That is precisely why multi-stop stabilisation claims require 1/4 s test exposures.

Caveat: the transition is not a cliff, and blur tolerance depends on output resolution, viewing size, subject contrast and sharpening. The 1–2 px threshold is conventional and I found no single authoritative numeric citation for it in these sources, so treat it as UNCERTAIN and note that all the figures above scale linearly with whichever threshold is chosen.

---

## 5. Rotation-to-pixel relationship, worked for 512 px wide

### 5.1 Formula
For a camera rotation θ about an axis perpendicular to the optical axis, the image displacement in pixels is

```
Δx_px = f_px · tan θ        with   f_px = N_px · f / W_sensor
```

- `f` = focal length (use 35 mm-equivalent and W_sensor = 36 mm for a full-frame-equivalent angular scale),
- `N_px` = frame width in pixels (512 here),
- equivalently `Δx_px = f_mm · tan θ · N_px / W_sensor`.

Pixels per degree (small-angle): **`k = N_px · f / W_sensor`** — notably **independent of pixel pitch**, and inversely proportional to focal length: `k = 512 · f / 36 = 14.222 · f`.

This is the same relation CIPA writes the other way round ("theoretical handheld blur amount [μm] = 35 mm-equivalent focal length f [mm] × tan θ × 1000") and the same one Canon's patent uses to convert "movable amount [mm] into correctable angle [deg]" via "focal length information". Sources: CIPA §4-5-1 (link in §1.1); https://patents.google.com/patent/US20200084358A1/en

### 5.2 Worked example — 512 px wide, 36 mm-equivalent frame width
DERIVED, `k = 512·f/36 = 14.222·f` px/deg; `Δx_px = f·tan θ·512/36`.

| Focal length (35 mm-equiv) | k (px per degree) | 0.1 ° | 0.5 ° | 1.0 ° |
|---|---|---|---|---|
| **24 mm** | 341.33 px/° | **34.1 px** | **170.7 px** | **341.3 px** |
| **40 mm** | 568.89 px/° | **56.9 px** | **284.4 px** | **568.9 px** |
| **85 mm** | 1208.89 px/° | **120.9 px** | **604.4 px** | **1208.9 px** |

Sanity notes: at 512 px wide, **0.5 ° at 40 mm moves 284 px = 55 % of the frame width**; **1.0 ° at 85 mm moves 1209 px = 2.4 frame widths**. At 512 px output, therefore, *sub-0.01 ° precision is meaningless* (0.01 ° = 12 px at 85 mm) — the pixel grid quantises the shake long before the physics does.

Scaling for other frame widths (DERIVED, same relation): a 1920 px-wide frame has k = 1920·f/36 = 53.33·f px/° — i.e. **3.75× the 512-px figures** for the same focal length and angle.

---

## Consolidated amplitude / frequency table

| # | Quantity | Value | Unit | Label | Source URL |
|---|---|---|---|---|---|
| 1 | Handheld still shake, average vibration angle, slowest tested exposure (1/4 s) | 1.06 – 3.36 | degrees | MEASURED-derived | https://www.cipa.jp/e/std/image-stabilization/documents/CIPA_DC-X011-2024_E.pdf |
| 2 | same, at 1/60 s | 0.019 – 0.039 | degrees | MEASURED-derived | same |
| 3 | same, at 1/2000 s | 0.0009 – 0.0034 | degrees | MEASURED-derived | same |
| 4 | Implied handheld angular rate, stills (all exposures) | ≈1 – 3.4 | deg/s | DERIVED | same |
| 5 | CIPA vibratory-apparatus calibration angular velocity | 6.28 (rows b–e); 1.26 printed for row a (inconsistent) | deg/s | SPEC + UNCERTAIN | https://www.cipa.jp/e/std/image-stabilization/documents/VMVA_for_DC-X011-2024_E.pdf |
| 6 | CIPA apparatus frequency coverage | 0.1, 0.5, 1, 5, 10 | Hz | SPEC | same |
| 7 | CIPA apparatus amplitudes | 2, 2, 1, 0.2, 0.1 | degrees (±) | SPEC | same |
| 8 | CIPA roll-axis test sine, 1 Hz | ±0.5 and ±0.2 | degrees | SPEC | same (Table 6-2-3) |
| 9 | Panasonic measured mean yaw shake, 1/8 s, one-handed, n>100, 10 subjects | 0.11 (mean), up to ≈0.42 (distribution) | degrees | MEASURED | https://www.jstage.jst.go.jp/article/itetr/34.1/0/34.1_39/_pdf/-char/ja |
| 10 | Panasonic, gyro-derived rate | 0.88 | deg/s | DERIVED | same |
| 11 | Panasonic translational shake, mean horizontal | 0.59 | mm | MEASURED | same |
| 12 | Panasonic yaw/roll vs pitch asymmetry | ≥30 | % | MEASURED | same |
| 13 | Xiao 2006 shake power-law exponent b | 0.5632 | — | MEASURED fit | https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2006/ICIS-0-736/33685.pdf |
| 14 | Xiao 2006 amplitude coefficient `a`, long axis | 0.092 (1251 g) – 0.1553 (335 g, non-expert) | degrees | MEASURED | same |
| 15 | Xiao 2006 σ at 1/50 s | 0.0102 – 0.0172 | degrees | DERIVED | same |
| 16 | Xiao 2006 equivalent rate at 1/50 s | 0.51 – 0.86 | deg/s | DERIVED | same |
| 17 | Xiao 2006 angular sampling resolution of test cameras | 0.0063 – 0.0068 | deg/pixel | MEASURED | same |
| 18 | DXOMARK walking angular position | ±2 to ±3 | degrees | MEASURED | https://corp.dxomark.com/2026/04/09/stab-series-how-to-create-and-use-realistic-movements-for-stabilization-testing-3-3/ |
| 19 | DXOMARK walking peak angular velocity | 10 – 20 | deg/s | MEASURED | same |
| 20 | DXOMARK motion energy concentration | below 10 | Hz | MEASURED | same |
| 21 | Toshiba/NHK 1987: shake needing correction | ≥1, correction band upper limit 10, "actually concentrated at a few Hz" | Hz | MEASURED | https://www.jstage.jst.go.jp/article/ncrtv/23/0/23_377/_pdf/-char/ja |
| 22 | Toshiba/NHK 1987 correction accuracy | 20 TV lines → ≈1 TV line (95 %) | TV lines | MEASURED | same |
| 23 | Canon patent: camera shake frequency | 1 – 12 | Hz | SPEC | https://patents.google.com/patent/JP2008139640A/en |
| 24 | Patent folklore: max shake amplitude | 0.2 | degrees | UNCERTAIN (contradicted) | https://patents.google.com/patent/JP2008139640A/en |
| 25 | Samsung Electro-Mechanics OIS correction angle | 3.0 (was 1.5) | degrees | SPEC | https://samsungsem.com/jp/newsroom/news/view.do?id=6401 |
| 26 | Cambridge Mechatronics OIS compensation angle | ±1.0 → ±1.5 | degrees | SPEC | https://www.cambridgemechatronics.com/cn/news/resources-blogs/increasing-image-stabilisation-50-stroke-extension/ |
| 27 | Cambridge Mechatronics test shake amplitude / frequency | 1.5 deg @ 6 Hz | deg, Hz | SPEC (test condition) | same |
| 28 | DeepOIS claim: OIS range | 1 – 2 | degrees | UNCERTAIN | http://liushuaicheng.org/CSVT/DeepOIS.pdf |
| 29 | Whyte IJCV 2012 blur-kernel angular support | [−5, +5] i.e. "a few degrees" | degrees | SPEC (model bound) | https://www.robots.ox.ac.uk/~vgg/publications/2012/Whyte12/whyte12.pdf |
| 30 | GyroDeblurNet gyro sampling rate | 200 | Hz | MEASURED | https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Gyro-based_Neural_Single_Image_Deblurring_CVPR_2025_paper.pdf |
| 31 | GyroDeblurNet capture / exposure | 195 s & 60 s; 39,116 & 12,065 samples; 1/20 s | s, samples, s | MEASURED | same |
| 32 | DeepGyro gyro sampling / exposure | 100 Hz; 30 ms | Hz, s | MEASURED | https://arxiv.org/pdf/1810.00986 |
| 33 | Rotation→pixels, 512 px wide, 24 mm | 341.33 px/°; 34.1 / 170.7 / 341.3 px at 0.1/0.5/1.0° | px | DERIVED | §5 |
| 34 | same, 40 mm | 568.89 px/°; 56.9 / 284.4 / 568.9 px | px | DERIVED | §5 |
| 35 | same, 85 mm | 1208.89 px/°; 120.9 / 604.4 / 1208.9 px | px | DERIVED | §5 |
| 36 | Blur from a 1.0 ° total rotation during a 1/50 s exposure, 512 px (24/40/85 mm) | 5.96 / 9.93 / 21.10 | px | DERIVED | §4.3 |
| 37 | Angular rate giving exactly 1 px blur at 1/50 s, 512 px (24/40/85 mm) | 8.39 / 5.04 / 2.37 | deg/s | DERIVED | §4.2 |
| 38 | Total rotation during a 1/50 s exposure that stays under 1 px at all three focal lengths | ≤0.05 | degrees | DERIVED | §4.3 |
| 39 | CIPA roll-component average vibration angle, 1/4 s / 1/8 s / 1/60 s | 0.09–2.03 / 0.06–0.09 / 0.009–0.020 | degrees | MEASURED-derived | §1.3 |
| 40 | CIPA roll as a fraction of yaw/pitch amplitude, all exposures | 0.55 – 0.62 | ratio | DERIVED | §1.3 |

---

## Where sources disagree

1. **Correctable angle definitions.** Samsung "3.0 degrees" (total or ±?) vs CML "±1.0 → ±1.5 degrees" vs DeepOIS "1–2 degrees". Not reconcilable without Samsung's datasheet; do not average them.
2. **CIPA Fig. 4-5-1c/d axis units.** `[°]` scale reaching 3.5 vs arcminute labels reaching 30′ (=0.5°). ~120× apart. I used `[°]`; flagged.
3. **CIPA VMVA Table 4-2-1a row a.** Printed (0.1 Hz, 2°, 1.26 deg/s) is inconsistent with rows b–e, which all encode 6.28 deg/s. One of the three printed values in row a is wrong.
4. **Max shake amplitude: 0.2° (patent folklore) vs 0.42° measured (Panasonic) vs 3.36° (CIPA envelope).** The 0.2° figure cannot be a ceiling on total angular excursion.
5. **"Stops" vs angle.** Marketing "N stops" and CIPA "stops" carry no angular information; there is no published mapping from stops to a correctable angle. Any such conversion is an extra assumption, not a citation.
6. **Focal-length rule of thumb.** The 1/f shutter rule ("a 50 mm lens needs ≤ 1/50 s") has no stated blur criterion behind it. Under the 1 px / 512 px criterion worked in §4, the rule is roughly right at 40–85 mm-equivalent and over-conservative at 24 mm. It is a preference statement, not a derived blur budget.

## Gaps (stated explicitly, not filled by guessing)
- No published **maximum correctable angular velocity in deg/s** for any OIS/IBIS system.
- No published **RMS angular velocity in deg/s** for any gyro shake dataset (GyroBlur, GoPro-Gyro, DeepGyro, RSBlur). Only sample-rate/duration/exposure metadata.
- **CIPA WB-H / WB-L waveform files** are member-walled; their own RMS/peak statistics are not publicly citable — only the derived average-vibration-angle envelope and the implied ~1–3.4 deg/s.
- No CIPA **video** image-stabilisation measurement standard exists on the public standards page.
- A **1 px / 2 px "visible blur" threshold** has no single authoritative numeric citation in the sources gathered.

---

## URLs

Standards
1. https://www.cipa.jp/e/std/image-stabilization/documents/CIPA_DC-X011-2024_E.pdf
2. https://www.cipa.jp/e/std/image-stabilization/documents/VMVA_for_DC-X011-2024_E.pdf
3. https://www.cipa.jp/e/std/image-stabilization.html

Manufacturer / industry
4. https://samsungsem.com/jp/newsroom/news/view.do?id=6401
5. https://zdnet.co.kr/view/?no=20230320094932
6. https://www.cambridgemechatronics.com/cn/news/resources-blogs/increasing-image-stabilisation-50-stroke-extension/
7. https://corp.dxomark.com/2026/04/09/stab-series-how-to-create-and-use-realistic-movements-for-stabilization-testing-3-3/
8. https://www.ricoh-imaging.co.jp/english/products/k-3-2/feature/02.html

Measured shake studies
9. https://www.jstage.jst.go.jp/article/itetr/34.1/0/34.1_39/_pdf/-char/ja
10. https://www.imaging.org/common/uploaded%20files/pdfs/Papers/2006/ICIS-0-736/33685.pdf
11. https://www.jstage.jst.go.jp/article/ncrtv/23/0/23_377/_pdf/-char/ja

Datasets / deblurring / stabilisation papers
12. https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Gyro-based_Neural_Single_Image_Deblurring_CVPR_2025_paper.pdf
13. https://arxiv.org/pdf/1810.00986
14. https://arxiv.org/pdf/1904.01594
15. https://arxiv.org/pdf/2204.00515
16. https://arxiv.org/pdf/2104.00317
17. https://www.robots.ox.ac.uk/~vgg/publications/2012/Whyte12/whyte12.pdf
18. https://www.di.ens.fr/willow/research/deblurring/poster_cvpr10.pdf
19. https://isy.gitlab-pages.liu.se/cvl/en/datasets/gopro-gyro-dataset/
20. http://liushuaicheng.org/CSVT/DeepOIS.pdf

Patents
21. https://patents.google.com/patent/US20200084358A1/en
22. https://patents.google.com/patent/JP2008139640A/en
23. https://patentimages.storage.googleapis.com/72/2b/1f/722bb6c77e1ab3/JP2008139640A.pdf
24. https://patents.google.com/patent/WO2019190159A1
