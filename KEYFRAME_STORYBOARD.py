"""
KEYFRAME GENERATION STORYBOARD
Visualizing the dense anchoring strategy WITHOUT T+1 interpolation
"""

# Example: User provides idea "A person discovers a mysterious well in an abandoned village"

# =============================================================================
# PHASE 1: OLLAMA DIRECTOR (Script Generation)
# =============================================================================

IDEA = "A person discovers a mysterious well in an abandoned village"

# Ollama generates script with dialogue/narration
SCRIPT = [
    {"speaker": "Narrator", "text": "The village had been forgotten for decades."},
    {"speaker": "Narrator", "text": "Jamie walked down the dusty main street."},
    {"speaker": "Narrator", "text": "At the edge of town stood an ancient well."},
    {"speaker": "Narrator", "text": "Something moved in the darkness below."}
]

# =============================================================================
# PHASE 2: VOICE MODULE (Audio + Timing)
# =============================================================================

# edge-tts generates audio for each line and measures duration
SCRIPT_WITH_TIMING = [
    {"speaker": "Narrator", "text": "The village had been forgotten for decades.", 
     "duration": 3.2, "audio": "line_001.mp3"},
    {"speaker": "Narrator", "text": "Jamie walked down the dusty main street.", 
     "duration": 2.8, "audio": "line_002.mp3"},
    {"speaker": "Narrator", "text": "At the edge of town stood an ancient well.", 
     "duration": 2.5, "audio": "line_003.mp3"},
    {"speaker": "Narrator", "text": "Something moved in the darkness below.", 
     "duration": 2.3, "audio": "line_004.mp3"}
]

# =============================================================================
# PHASE 3: SHOT PLANNING (Visual Description)
# =============================================================================

# Ollama creates detailed visual prompts for each audio segment
SHOTS = [
    {
        "duration": 3.2,
        "subject": "abandoned village",
        "location": "desert village street",
        "camera": "wide establishing shot, static",
        "lighting": "harsh midday sun, long shadows",
        "prompt": "Wide shot of abandoned village street, crumbling mud-brick buildings, dusty ground, harsh sunlight casting long shadows, deserted atmosphere, filmic grain, muted colors",
        "audio": "line_001.mp3"
    },
    {
        "duration": 2.8,
        "subject": "person walking",
        "location": "village street",
        "camera": "tracking medium shot, follows subject",
        "lighting": "natural daylight, dust particles visible",
        "prompt": "Medium tracking shot of person walking down dusty village street, worn clothing, cautious movement, abandoned buildings on sides, natural lighting, observational framing",
        "audio": "line_002.mp3"
    },
    {
        "duration": 2.5,
        "subject": "ancient stone well",
        "location": "village edge",
        "camera": "slow approach, low angle",
        "lighting": "soft light, shadows around well opening",
        "prompt": "Low angle shot of ancient stone well at village edge, weathered stones, dark opening, surrounding desert landscape, ominous feeling, textured surfaces, soft natural light",
        "audio": "line_003.mp3"
    },
    {
        "duration": 2.3,
        "subject": "well darkness",
        "location": "looking into well",
        "camera": "close up, peering down",
        "lighting": "darkness with hint of movement",
        "prompt": "Close shot looking down into dark well shaft, barely visible depths, suggestion of movement in shadows, stone texture, mysterious atmosphere, grainy low light",
        "audio": "line_004.mp3"
    }
]

# =============================================================================
# PHASE 4: KEYFRAME GENERATION (Dense Anchoring)
# =============================================================================

ANCHOR_INTERVAL = 1.0  # Generate Teacher keyframe every 1 second

"""
SHOT 1: "Abandoned village street" (3.2 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Keyframes needed: ceil(3.2 / 1.0) + 1 = 4 keyframes
Timestamps: 0.0s, 1.0s, 2.0s, 3.2s

[KF 0] @ 0.0s
  Generation: FRESH (MobileDiffusion text-to-image)
  Prompt: "Wide shot of abandoned village street, crumbling mud-brick buildings..."
  → shot_00_kf00.png
  
[KF 1] @ 1.0s  
  Generation: IMG2IMG from KF 0 (strength=0.35)
  Prompt: Same prompt
  Purpose: Slight camera movement, maintain scene continuity
  → shot_00_kf01.png
  
[KF 2] @ 2.0s
  Generation: IMG2IMG from KF 1 (strength=0.35)
  Prompt: Same prompt
  Purpose: Continue camera movement, consistent lighting/color
  → shot_00_kf02.png
  
[KF 3] @ 3.2s
  Generation: IMG2IMG from KF 2 (strength=0.35)
  Prompt: Same prompt
  Purpose: End frame of shot, ready for transition
  → shot_00_kf03.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

"""
SHOT 2: "Person walking down street" (2.8 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Keyframes needed: ceil(2.8 / 1.0) + 1 = 3 keyframes
Timestamps: 0.0s, 1.0s, 2.8s

[KF 0] @ 0.0s
  Generation: IMG2IMG from previous shot's last keyframe (strength=0.35)
  Prompt: "Medium tracking shot of person walking down dusty village street..."
  Purpose: Smooth transition from Shot 1 to Shot 2
  → shot_01_kf00.png
  
[KF 1] @ 1.0s
  Generation: IMG2IMG from KF 0 (strength=0.35)
  Prompt: Same prompt
  Purpose: Person has moved further down street
  → shot_01_kf01.png
  
[KF 2] @ 2.8s
  Generation: IMG2IMG from KF 1 (strength=0.35)
  Prompt: Same prompt
  Purpose: Person approaching end of street
  → shot_01_kf02.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

"""
SHOT 3: "Ancient stone well" (2.5 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Keyframes needed: ceil(2.5 / 1.0) + 1 = 3 keyframes
Timestamps: 0.0s, 1.0s, 2.5s

[KF 0] @ 0.0s
  Generation: IMG2IMG from previous shot's last keyframe (strength=0.35)
  Prompt: "Low angle shot of ancient stone well at village edge..."
  Purpose: Transition to well location
  → shot_02_kf00.png
  
[KF 1] @ 1.0s
  Generation: IMG2IMG from KF 0 (strength=0.35)
  Prompt: Same prompt
  Purpose: Camera slowly approaching well
  → shot_02_kf01.png
  
[KF 2] @ 2.5s
  Generation: IMG2IMG from KF 1 (strength=0.35)
  Prompt: Same prompt
  Purpose: Close to well, ready for next shot
  → shot_02_kf02.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

"""
SHOT 4: "Looking into well darkness" (2.3 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Keyframes needed: ceil(2.3 / 1.0) + 1 = 3 keyframes
Timestamps: 0.0s, 1.0s, 2.3s

[KF 0] @ 0.0s
  Generation: IMG2IMG from previous shot's last keyframe (strength=0.35)
  Prompt: "Close shot looking down into dark well shaft..."
  Purpose: Transition to peering into well
  → shot_03_kf00.png
  
[KF 1] @ 1.0s
  Generation: IMG2IMG from KF 0 (strength=0.35)
  Prompt: Same prompt
  Purpose: Focus deeper into darkness
  → shot_03_kf01.png
  
[KF 2] @ 2.3s
  Generation: IMG2IMG from KF 1 (strength=0.35)
  Prompt: Same prompt
  Purpose: Final frame, suggestion of movement
  → shot_03_kf02.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# KEYFRAME GENERATION SUMMARY
# =============================================================================

TOTAL_KEYFRAMES = 4 + 3 + 3 + 3  # 13 keyframes
TOTAL_DURATION = 3.2 + 2.8 + 2.5 + 2.3  # 10.8 seconds

print(f"""
KEYFRAME-ONLY STORYBOARD SUMMARY
═════════════════════════════════════════════════════════════════

Total Video Duration: {TOTAL_DURATION}s
Total Keyframes Generated: {TOTAL_KEYFRAMES}
Average Keyframe Spacing: {TOTAL_DURATION / (TOTAL_KEYFRAMES - 1):.2f}s

Shot 1: 4 keyframes (0s, 1s, 2s, 3.2s)
Shot 2: 3 keyframes (0s, 1s, 2.8s)  
Shot 3: 3 keyframes (0s, 1s, 2.5s)
Shot 4: 3 keyframes (0s, 1s, 2.3s)

═════════════════════════════════════════════════════════════════

VISUAL CONSISTENCY STRATEGY:

1st keyframe of each shot:
  - Shot 1 KF0: FRESH generation (establishes scene)
  - Shot 2 KF0: IMG2IMG from Shot 1 KF3 (smooth transition)
  - Shot 3 KF0: IMG2IMG from Shot 2 KF2 (smooth transition)
  - Shot 4 KF0: IMG2IMG from Shot 3 KF2 (smooth transition)

Subsequent keyframes within shot:
  - IMG2IMG from previous keyframe (strength=0.35)
  - Maintains visual continuity
  - Allows gradual camera movement/changes
  - Prevents sudden jumps in color/lighting/structure

═════════════════════════════════════════════════════════════════

WHAT HAPPENS BETWEEN KEYFRAMES (WITH T+1):

Between each pair of keyframes (e.g., KF0 → KF1):
  - Duration: 1.0 second
  - Frames needed: 24 frames @ 24fps
  - T+1 VAE generates 24 intermediate frames
  - SR Upscaler restores details (64x64 → 128x128)
  - Result: Smooth motion between anchors

WITHOUT T+1 (keyframes only):
  - You'd see 13 still images with cuts between them
  - Each image holds for ~1 second
  - Like a slideshow with transitions
  - No motion, just sequence of poses

═════════════════════════════════════════════════════════════════
""")

# =============================================================================
# KEYFRAME GENERATION PARAMETERS
# =============================================================================

GENERATION_SETTINGS = {
    "teacher_model": "MobileDiffusion (tiny-sd)",
    "resolution": "128x128",
    "num_inference_steps": 15,
    "guidance_scale": 7.0,
    "img2img_strength": 0.35,
    "anchor_interval": 1.0,
    "fps": 24,
}

print("GENERATION SETTINGS:")
for key, value in GENERATION_SETTINGS.items():
    print(f"  {key}: {value}")
