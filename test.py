"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SEMAPHORE TO KEYBOARD                                ║
║                          CONFIGURATION INDEX                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TUNING THRESHOLDS (most likely to need adjustment)                          ║
║  ─────────────────────────────────────────────────                           ║
║  VISIBILITY_THRESHOLD      line 46   0.0–1.0. Lower = accept less-confident ║
║                                      landmark detections. Raise if ghost     ║
║                                      arms are triggering false positives.    ║
║                                                                              ║
║  STRAIGHT_LIMB_MARGIN      line 47   Degrees of bend allowed before an arm  ║
║                                      is no longer "straight". Raise if your  ║
║                                      arms are rarely detected as pointing.   ║
║                                                                              ║
║  EXTENDED_LIMB_MARGIN      line 48   Ratio: forearm must be at least this    ║
║                                      fraction of upper-arm length. Raise if  ║
║                                      bent elbows are triggering matches.     ║
║                                                                              ║
║  SNAP_TOLERANCE            line 49   Max degrees of error (per arm) allowed  ║
║                                      when snapping raw angles to the nearest ║
║                                      valid semaphore. Raise to be more       ║
║                                      forgiving; lower for stricter matching. ║
║                                                                              ║
║  GESTURE_CONFIRMATION_FRAMES line 50 How many consecutive video frames must  ║
║                                      show the same pose before it fires.     ║
║                                      Raise to reduce accidental triggers;    ║
║                                      lower for faster response.              ║
║                                                                              ║
║  ANGLE_SNAP_STEP           line 51   Granularity of angle rounding (degrees).║
║                                      Keep at 45 to match the semaphore table.║
║                                                                              ║
║  JUMP DETECTION (caps lock trigger)                                          ║
║  ──────────────────────────────────                                          ║
║  JUMP_THRESHOLD            line 53   How far (0.0–1.0, normalised to frame  ║
║                                      height) the hips must rise above the   ║
║                                      rolling baseline to count as a jump.   ║
║                                      Lower = easier to trigger; raise if    ║
║                                      normal movement is causing false fires. ║
║                                                                              ║
║  JUMP_BASELINE_FRAMES      line 54   How many recent frames to average for  ║
║                                      the "standing" hip baseline. Larger =  ║
║                                      slower to adapt to the performer       ║
║                                      moving towards/away from camera.       ║
║                                                                              ║
║  JUMP_COOLDOWN_FRAMES      line 55   Frames to ignore after a jump fires,   ║
║                                      preventing one jump from toggling caps  ║
║                                      lock multiple times.                   ║
║                                                                              ║
║  BOW DETECTION (tap ↔ hold mode toggle)                                     ║
║  ──────────────────────────────────────                                      ║
║  BOW_THRESHOLD             line 57   How much the shoulder-to-hip distance  ║
║                                      must shrink (normalised 0–1) below the ║
║                                      baseline to count as a bow. Lower =    ║
║                                      more sensitive; raise if normal         ║
║                                      posture changes are false-triggering.  ║
║                                                                              ║
║  BOW_BASELINE_FRAMES       line 58   Frames of torso-height history used to ║
║                                      compute the upright baseline. Larger = ║
║                                      slower to adapt to camera distance.    ║
║                                                                              ║
║  BOW_COOLDOWN_FRAMES       line 59   Frames to suppress re-triggering after ║
║                                      a bow fires (~1 s at 30 fps).          ║
║                                                                              ║
║  SPAM_RATE                 line 61   Key presses per second in spam mode.   ║
║                                      Only used for keys in TAP_REPEAT_KEYS. ║
║                                      All other keys are held continuously.  ║
║                                                                              ║
║  SPAM_DEBOUNCE_FRAMES      line 62   Consecutive frames of no detected pose ║
║                                      before hold stops. Stops a single      ║
║                                      wobbly frame breaking the hold.        ║
║                                                                              ║
║  OUTPUT                                                                      ║
║  ──────                                                                      ║
║  OUTPUT_FILE               line 54   Path of the text file that confirmed    ║
║                                      gestures are written to. Set to None    ║
║                                      to disable file output.                 ║
║                                                                              ║
║  CAMERA / MODEL                                                              ║
║  ──────────────                                                              ║
║  MODEL_PATH                line 57   Path to the MediaPipe pose model file.  ║
║  CAMERA_INDEX              line 58   0 = default webcam, 1 = next camera.   ║
║                                                                              ║
║  CLI FLAGS                                                                   ║
║  ─────────                                                                   ║
║  --type  /  -t             Actually send keypresses (default: display only)  ║
║  --tolerance N             Override SNAP_TOLERANCE at runtime                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import ctypes
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist
from math import atan2, degrees

# ── TUNING THRESHOLDS ─────────────────────────────────────────────────────────
VISIBILITY_THRESHOLD = 0.5
STRAIGHT_LIMB_MARGIN = 20
EXTENDED_LIMB_MARGIN = 0.8
SNAP_TOLERANCE = 15
GESTURE_CONFIRMATION_FRAMES = 5
ANGLE_SNAP_STEP = 45

# ── JUMP DETECTION ────────────────────────────────────────────────────────────
JUMP_THRESHOLD = 0.06
JUMP_BASELINE_FRAMES = 20
JUMP_COOLDOWN_FRAMES = 30

# ── BOW DETECTION ─────────────────────────────────────────────────────────────
BOW_THRESHOLD = 0.08
BOW_BASELINE_FRAMES = 20
BOW_COOLDOWN_FRAMES = 40

# ── HOLD / SPAM MODE ──────────────────────────────────────────────────────────
SPAM_RATE = 10
SPAM_DEBOUNCE_FRAMES = 8

# Keys tapped repeatedly rather than truly held down.
TAP_REPEAT_KEYS = {"escape", "tab", "enter", "backspace", "capslock"}

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "semaphore_output.txt"

# ── CAMERA / MODEL ────────────────────────────────────────────────────────────
MODEL_PATH = 'pose_landmarker_heavy.task'
CAMERA_INDEX = 0

# ── SKELETON CONNECTIONS ──────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# ── SEMAPHORE LOOKUP TABLE ────────────────────────────────────────────────────
SEMAPHORES = {
    (-90, -45): {'a': "a",       'n': "1"},
    (-90,   0): {'a': "b",       'n': "2"},
    (-90,  45): {'a': "c",       'n': "3"},
    (-90,  90): {'a': "d",       'n': "4"},
    (135, -90): {'a': "e",       'n': "5"},
    (180, -90): {'a': "f",       'n': "6"},
    (225, -90): {'a': "g",       'n': "7"},
    (-45,   0): {'a': "h",       'n': "8"},
    (-45,  45): {'a': "i",       'n': "9"},
    (180,  90): {'a': "j",       'n': "capslock"},
    (90, -45): {'a': "k",       'n': "0"},
    (135, -45): {'a': "l",       'n': "\\"},
    (180, -45): {'a': "m",       'n': "["},
    (225, -45): {'a': "n",       'n': "]"},
    (0,  45): {'a': "o",       'n': ","},
    (90,   0): {'a': "p",       'n': ";"},
    (135,   0): {'a': "q",       'n': "="},
    (180,   0): {'a': "r",       'n': "-"},
    (225,   0): {'a': "s",       'n': "."},
    (90,  45): {'a': "t",       'n': "`"},
    (135,  45): {'a': "u",       'n': "/"},
    (225,  90): {'a': "v",       'n': '"'},
    (135, 180): {'a': "w"},
    (135, 225): {'a': "x",       'n': ""},
    (180,  45): {'a': "y"},
    (180, 225): {'a': "z"},
    (90,  90): {'a': "space",   'n': "enter"},
    (135,  90): {'a': "tab"},
    (225,  45): {'a': "escape"},
}

# ── STATE ─────────────────────────────────────────────────────────────────────
gesture_buffer = []
last_confirmed_gesture = None
caps_lock_active = False
hip_y_history = []
jump_cooldown = 0
hold_mode_active = False
spam_key = None
last_spam_time = 0.0
spam_debounce_counter = 0
torso_height_history = []
bow_cooldown = 0


# ── DRAWING HELPER: semi-transparent filled rect ──────────────────────────────

def _fill_rect_alpha(image, x1, y1, x2, y2, color_bgr, alpha=0.45):
    """
    Blend a filled rectangle onto `image` in-place.
    alpha=0.0 → invisible, alpha=1.0 → fully opaque.
    Clamps coordinates to valid image bounds automatically.
    """
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), color_bgr, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


# ── GEOMETRY HELPERS ──────────────────────────────────────────────────────────

def get_angle(a, b, c):
    ang = degrees(
        atan2(c['y'] - b['y'], c['x'] - b['x']) -
        atan2(a['y'] - b['y'], a['x'] - b['x'])
    )
    return ang + 360 if ang < 0 else ang


def get_limb_direction(arm):
    dy = arm[2]['y'] - arm[0]['y']
    dx = arm[2]['x'] - arm[0]['x']
    angle = degrees(atan2(dy, dx))
    mod_close = angle % ANGLE_SNAP_STEP
    angle -= mod_close
    if mod_close > ANGLE_SNAP_STEP / 2:
        angle += ANGLE_SNAP_STEP
    angle = int(angle)
    return -90 if angle == 270 else angle


def is_limb_pointing(upper, mid, lower):
    if any(j['visibility'] < VISIBILITY_THRESHOLD for j in [upper, mid, lower]):
        return False
    if abs(180 - get_angle(upper, mid, lower)) >= STRAIGHT_LIMB_MARGIN:
        return False
    u_len = dist.euclidean([upper['x'], upper['y']], [mid['x'], mid['y']])
    l_len = dist.euclidean([lower['x'], lower['y']], [mid['x'], mid['y']])
    return l_len >= EXTENDED_LIMB_MARGIN * u_len


# ── JUMP DETECTION ────────────────────────────────────────────────────────────

def detect_jump(body):
    global hip_y_history, jump_cooldown
    if jump_cooldown > 0:
        jump_cooldown -= 1
    hipL, hipR = body[23], body[24]
    if hipL['visibility'] < VISIBILITY_THRESHOLD or hipR['visibility'] < VISIBILITY_THRESHOLD:
        return False
    current_hip_y = (hipL['y'] + hipR['y']) / 2.0
    hip_y_history.append(current_hip_y)
    if len(hip_y_history) > JUMP_BASELINE_FRAMES:
        hip_y_history.pop(0)
    if len(hip_y_history) < JUMP_BASELINE_FRAMES:
        return False
    baseline = sum(hip_y_history[:-3]) / len(hip_y_history[:-3])
    if current_hip_y > baseline + JUMP_THRESHOLD and jump_cooldown == 0:
        jump_cooldown = JUMP_COOLDOWN_FRAMES
        return True
    return False


# ── BOW DETECTION ─────────────────────────────────────────────────────────────

def detect_bow(body):
    global torso_height_history, bow_cooldown
    if bow_cooldown > 0:
        bow_cooldown -= 1
    shoulderL, shoulderR = body[11], body[12]
    hipL,      hipR = body[23], body[24]
    if any(lm['visibility'] < VISIBILITY_THRESHOLD for lm in [shoulderL, shoulderR, hipL, hipR]):
        return False
    current_torso_height = ((shoulderL['y'] + shoulderR['y']) / 2.0 -
                            (hipL['y'] + hipR['y']) / 2.0)
    torso_height_history.append(current_torso_height)
    if len(torso_height_history) > BOW_BASELINE_FRAMES:
        torso_height_history.pop(0)
    if len(torso_height_history) < BOW_BASELINE_FRAMES:
        return False
    baseline = sum(torso_height_history[:-3]) / len(torso_height_history[:-3])
    if current_torso_height < baseline - BOW_THRESHOLD and bow_cooldown == 0:
        bow_cooldown = BOW_COOLDOWN_FRAMES
        return True
    return False


# ── SEMAPHORE MATCHING ────────────────────────────────────────────────────────

def snap_to_nearest_semaphore(detected_l_ang, detected_r_ang, tolerance=None):
    if tolerance is None:
        tolerance = SNAP_TOLERANCE
    best_match, best_distance = None, float('inf')
    for (l_ang, r_ang), semaphore in SEMAPHORES.items():
        l_diff = min(abs(detected_l_ang - l_ang),
                     360 - abs(detected_l_ang - l_ang))
        r_diff = min(abs(detected_r_ang - r_ang),
                     360 - abs(detected_r_ang - r_ang))
        total = l_diff + r_diff
        if total < best_distance and total <= tolerance * 2:
            best_distance = total
            best_match = (l_ang, r_ang)
    if best_match:
        return best_match[0], best_match[1], SEMAPHORES[best_match]
    return None, None, None


# ── GESTURE CONFIRMATION ──────────────────────────────────────────────────────

def check_gesture_confirmation(detected_gesture):
    global gesture_buffer, last_confirmed_gesture
    if detected_gesture is None:
        gesture_buffer = []
        return None
    if gesture_buffer and gesture_buffer[-1] != detected_gesture:
        gesture_buffer = []
    gesture_buffer.append(detected_gesture)
    if len(gesture_buffer) >= GESTURE_CONFIRMATION_FRAMES:
        if detected_gesture != last_confirmed_gesture:
            last_confirmed_gesture = detected_gesture
            gesture_buffer = []
            return detected_gesture
    return None


# ── KEY HELPERS ───────────────────────────────────────────────────────────────

def _resolve_key(key, caps_active):
    if caps_active and len(key) == 1 and key.isalpha():
        return f"shift+{key}", key.upper()
    return key, key


def _press_key(kb_key, send_keypress):
    if send_keypress:
        import keyboard
        keyboard.press(kb_key)


def _release_key(kb_key, send_keypress):
    if send_keypress:
        import keyboard
        keyboard.release(kb_key)


def fire_keypress(kb_key, send_keypress):
    if send_keypress:
        import keyboard
        keyboard.press_and_release(kb_key)


def release_held_key(send_keypress):
    global spam_key, spam_debounce_counter, last_confirmed_gesture
    if spam_key is not None:
        if spam_key not in TAP_REPEAT_KEYS:
            _release_key(spam_key, send_keypress)
            print(f"  ✋ Released held key '{spam_key}'")
        else:
            print(f"  ✋ Stopped tap-repeat of '{spam_key}'")
        spam_key = None
    spam_debounce_counter = 0
    last_confirmed_gesture = None


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def output_gesture(key, frame, send_keypress, log_file, caps_active, hold_mode):
    global spam_key, last_spam_time
    from datetime import datetime

    kb_key, display_key = _resolve_key(key, caps_active)
    timestamp = datetime.now().strftime("%H:%M:%S")
    mode_tag = "[HOLD]" if hold_mode else "[TAP] "
    print(f"[{timestamp}] ✓ {mode_tag} OUTPUT: '{display_key}'")

    if log_file is not None:
        log_file.write(f"[{timestamp}] {mode_tag} {display_key}\n")
        log_file.flush()

    if hold_mode:
        if kb_key in TAP_REPEAT_KEYS:
            if spam_key != kb_key:
                if spam_key is not None:
                    _release_key(spam_key, send_keypress)
                spam_key = kb_key
                last_spam_time = 0.0
                print(f"  → Tap-repeat key set to '{kb_key}'")
        else:
            if spam_key != kb_key:
                if spam_key is not None:
                    _release_key(spam_key, send_keypress)
                    print(f"  → Released '{spam_key}'")
                spam_key = kb_key
                _press_key(kb_key, send_keypress)
                print(f"  → Holding '{kb_key}' down")
    else:
        fire_keypress(kb_key, send_keypress)
        if not send_keypress:
            cv2.putText(frame, display_key, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)


# ── PYGAME OVERLAY ────────────────────────────────────────────────────────────
# Transparent always-on-top window showing a 2D Minecraft paper-doll character
# whose limbs mirror the performer's pose in real time.
#
# CHROMA_KEY is rendered fully transparent by the OS (Windows layered window).
# Nothing in the drawing code should use exactly this colour.

CHROMA_KEY = (0, 0, 0)    # pure black → transparent
OVERLAY_W = 300           # overlay window width  — resize to taste
OVERLAY_H = 500           # overlay window height — resize to taste

# ── Minecraft-palette colours ─────────────────────────────────────────────────
MC_SKIN = (198, 160, 120)   # face / hands
MC_HAIR = (90,  60,  30)   # hair pixels on head
MC_SHIRT = (55, 100, 180)   # torso / upper arm (blue shirt)
MC_SHIRT_D = (40,  75, 140)   # darker shade for limb outlines
MC_TROUSER = (50,  50, 160)   # trousers
MC_TROUSER_D = (35,  35, 120)   # darker trouser outline
MC_BOOT = (80,  50,  20)   # boots
MC_OUTLINE = (20,  20,  20)   # near-black outline (not pure black!)

# ── HUD colours ───────────────────────────────────────────────────────────────
HUD_COLOR = (255, 220,   0)
CONFIRM_COLOR = (0, 255, 120)
WARN_COLOR = (255, 100,  60)


# ── GEOMETRY UTILITY ──────────────────────────────────────────────────────────

def _rotated_rect_points(cx, cy, w, h, angle_deg):
    """
    Return the four corners of a rectangle centred at (cx, cy),
    with size (w × h), rotated by angle_deg clockwise.
    Used to draw each blocky body segment.
    """
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw,  hh), (-hw,  hh)]
    return [
        (int(cx + cos_a * x - sin_a * y),
         int(cy + sin_a * x + cos_a * y))
        for x, y in corners
    ]


def _draw_block(surface, fill, outline, cx, cy, w, h, angle_deg):
    """Draw a rotated filled rectangle with an outline — one body segment."""
    pts = _rotated_rect_points(cx, cy, w, h, angle_deg)
    pygame.draw.polygon(surface, fill,    pts)
    pygame.draw.polygon(surface, outline, pts, 2)


def _lm(landmarks, idx):
    """Return (x, y) for a landmark scaled to the overlay window, with X mirrored."""
    lm = landmarks[idx]
    return (int((1 - lm.x) * OVERLAY_W),
            int(lm.y * OVERLAY_H))


def _angle_between(p1, p2):
    """
    Angle in degrees of the vector from p1 → p2, measured clockwise from
    straight down (so arms hanging = 90°, arm pointing right = 0°, etc.)
    We use clockwise-from-up so it matches pygame's coordinate system
    (y increases downward).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # note: arctan2(x, y) = clockwise from up
    return degrees(np.arctan2(dx, dy))


def _midpoint(p1, p2, t=0.5):
    """Return the point t-fraction of the way from p1 to p2."""
    return (p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t)


# ── MINECRAFT PAPER DOLL ──────────────────────────────────────────────────────

def _draw_minecraft_character(surface, landmarks):
    """
    Draw a blocky Minecraft-style 2D character driven by MediaPipe landmarks.

    Body parts and the landmark indices used to position / angle them:
      Head    — midpoint of shoulders, fixed upright
      Torso   — shoulder-mid → hip-mid
      U.Arms  — shoulder → elbow   (×2)
      Forearm — elbow → wrist      (×2)
      Thigh   — hip → knee         (×2)
      Shin    — knee → ankle       (×2)
    """
    if not landmarks:
        return

    # ── Gather key positions ──────────────────────────────────────────────────
    lsh = _lm(landmarks, 11)   # left  shoulder
    rsh = _lm(landmarks, 12)   # right shoulder
    lel = _lm(landmarks, 13)   # left  elbow
    rel = _lm(landmarks, 14)   # right elbow
    lwr = _lm(landmarks, 15)   # left  wrist
    rwr = _lm(landmarks, 16)   # right wrist
    lhip = _lm(landmarks, 23)   # left  hip
    rhip = _lm(landmarks, 24)   # right hip
    lkn = _lm(landmarks, 25)   # left  knee
    rkn = _lm(landmarks, 26)   # right knee
    lank = _lm(landmarks, 27)   # left  ankle
    rank = _lm(landmarks, 28)   # right ankle

    sh_mid = _midpoint(lsh,  rsh)
    hip_mid = _midpoint(lhip, rhip)

    # ── Derive body scale from shoulder width so it adapts to camera distance ─
    shoulder_w = max(abs(lsh[0] - rsh[0]), 20)   # pixels between shoulders
    scale = shoulder_w / 60.0               # 60 px is the "reference" width

    # Block sizes (width × height) at scale=1
    HEAD_W,  HEAD_H = int(50 * scale), int(50 * scale)
    TORSO_W, TORSO_H = int(54 * scale), int(72 * scale)
    UARM_W,  UARM_H = int(20 * scale), int(38 * scale)
    FARM_W,  FARM_H = int(18 * scale), int(36 * scale)
    THIGH_W, THIGH_H = int(22 * scale), int(40 * scale)
    SHIN_W,  SHIN_H = int(20 * scale), int(38 * scale)

    torso_angle = _angle_between(sh_mid, hip_mid)

    # ── Draw order: back limbs first, then torso, then front limbs ───────────
    # We treat the right side as "back" (drawn first) and left as "front".

    # ── Right leg (back) ──────────────────────────────────────────────────────
    rthigh_angle = _angle_between(rhip, rkn)
    rthigh_mid = _midpoint(rhip, rkn)
    _draw_block(surface, MC_TROUSER, MC_TROUSER_D,
                rthigh_mid[0], rthigh_mid[1], THIGH_W, THIGH_H, rthigh_angle)

    rshin_angle = _angle_between(rkn, rank)
    rshin_mid = _midpoint(rkn, rank)
    _draw_block(surface, MC_TROUSER, MC_BOOT,
                rshin_mid[0], rshin_mid[1], SHIN_W, SHIN_H, rshin_angle)

    # Boot (right)
    _draw_block(surface, MC_BOOT, MC_OUTLINE,
                rank[0], rank[1], SHIN_W, int(12 * scale), rshin_angle)

    # ── Right arm (back) ──────────────────────────────────────────────────────
    ruarm_angle = _angle_between(rsh, rel)
    ruarm_mid = _midpoint(rsh, rel)
    _draw_block(surface, MC_SHIRT, MC_SHIRT_D,
                ruarm_mid[0], ruarm_mid[1], UARM_W, UARM_H, ruarm_angle)

    rfarm_angle = _angle_between(rel, rwr)
    rfarm_mid = _midpoint(rel, rwr)
    _draw_block(surface, MC_SKIN, MC_SHIRT_D,
                rfarm_mid[0], rfarm_mid[1], FARM_W, FARM_H, rfarm_angle)

    # ── Torso ─────────────────────────────────────────────────────────────────
    torso_mid = _midpoint(sh_mid, hip_mid)
    _draw_block(surface, MC_SHIRT, MC_SHIRT_D,
                torso_mid[0], torso_mid[1], TORSO_W, TORSO_H, torso_angle)

    # ── Head ──────────────────────────────────────────────────────────────────
    # Position the head above the shoulder midpoint
    head_cx = int(sh_mid[0])
    head_cy = int(sh_mid[1] - HEAD_H * 0.65)
    # Face (skin)
    _draw_block(surface, MC_SKIN, MC_OUTLINE,
                head_cx, head_cy, HEAD_W, HEAD_H, 0)
    # Hair strip across top ~30% of head
    hair_h = int(HEAD_H * 0.35)
    hair_cy = head_cy - HEAD_H // 2 + hair_h // 2
    _draw_block(surface, MC_HAIR, MC_OUTLINE,
                head_cx, hair_cy, HEAD_W, hair_h, 0)
    # Eyes — two small dark squares
    eye_y = head_cy - int(HEAD_H * 0.08)
    eye_off = int(HEAD_W * 0.18)
    eye_size = max(int(HEAD_W * 0.14), 3)
    for ex in [head_cx - eye_off, head_cx + eye_off]:
        pygame.draw.rect(surface, (30, 30, 80),
                         (ex - eye_size // 2, eye_y - eye_size // 2,
                          eye_size, eye_size))

    # ── Left leg (front) ──────────────────────────────────────────────────────
    lthigh_angle = _angle_between(lhip, lkn)
    lthigh_mid = _midpoint(lhip, lkn)
    _draw_block(surface, MC_TROUSER, MC_TROUSER_D,
                lthigh_mid[0], lthigh_mid[1], THIGH_W, THIGH_H, lthigh_angle)

    lshin_angle = _angle_between(lkn, lank)
    lshin_mid = _midpoint(lkn, lank)
    _draw_block(surface, MC_TROUSER, MC_BOOT,
                lshin_mid[0], lshin_mid[1], SHIN_W, SHIN_H, lshin_angle)

    # Boot (left)
    _draw_block(surface, MC_BOOT, MC_OUTLINE,
                lank[0], lank[1], SHIN_W, int(12 * scale), lshin_angle)

    # ── Left arm (front) ──────────────────────────────────────────────────────
    luarm_angle = _angle_between(lsh, lel)
    luarm_mid = _midpoint(lsh, lel)
    _draw_block(surface, MC_SHIRT, MC_SHIRT_D,
                luarm_mid[0], luarm_mid[1], UARM_W, UARM_H, luarm_angle)

    lfarm_angle = _angle_between(lel, lwr)
    lfarm_mid = _midpoint(lel, lwr)
    _draw_block(surface, MC_SKIN, MC_SHIRT_D,
                lfarm_mid[0], lfarm_mid[1], FARM_W, FARM_H, lfarm_angle)


# ── WINDOW SETUP ──────────────────────────────────────────────────────────────

def _make_overlay_window():
    """
    Create and return a borderless, always-on-top, transparent pygame window.
    Works on Windows; on other platforms the window will appear but may not
    be transparent (it will just show a black background instead).
    """
    pygame.init()
    screen = pygame.display.set_mode((OVERLAY_W, OVERLAY_H), pygame.NOFRAME)
    pygame.display.set_caption("Semaphore Overlay")

    try:
        hwnd = pygame.display.get_wm_info()["window"]
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                            style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0x000000, 255, 1)
        HWND_TOPMOST = -1
        ctypes.windll.user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0, 0x0002 | 0x0001)
    except Exception:
        pass

    screen.set_colorkey(CHROMA_KEY)
    return screen


# ── MAIN DRAW CALL ────────────────────────────────────────────────────────────

def draw_overlay(screen, pose_landmarks, detected_gesture, buffer_count,
                 confirmed, caps_active, hold_mode, spam_key_name):
    """Render one frame of the transparent overlay."""
    screen.fill(CHROMA_KEY)

    font_big = pygame.font.SysFont("consolas", 32, bold=True)
    font_med = pygame.font.SysFont("consolas", 18, bold=True)
    font_sml = pygame.font.SysFont("consolas", 14)

    # ── Minecraft character ───────────────────────────────────────────────────
    _draw_minecraft_character(screen, pose_landmarks)

    # ── HUD — top-left corner ─────────────────────────────────────────────────
    y = 6
    if detected_gesture:
        display = (detected_gesture.upper()
                   if caps_active and len(detected_gesture) == 1 and detected_gesture.isalpha()
                   else detected_gesture)
        color = CONFIRM_COLOR if (hold_mode and spam_key_name) else HUD_COLOR
        surf = font_big.render(display, True, color)
        # Dark shadow so text is readable over the character
        shadow = font_big.render(display, True, (10, 10, 10))
        screen.blit(shadow, (10, y + 2))
        screen.blit(surf,   (8,  y))
        y += 40

        # Confirm progress bar
        ratio = min(buffer_count / GESTURE_CONFIRMATION_FRAMES, 1.0)
        bar_w = OVERLAY_W - 16
        bar_done = int(bar_w * ratio)
        bar_color = CONFIRM_COLOR if ratio >= 1.0 else WARN_COLOR
        pygame.draw.rect(screen, (40, 40, 40), (8, y, bar_w, 8))
        pygame.draw.rect(screen, bar_color,    (8, y, bar_done, 8))
        y += 14

        if confirmed:
            surf = font_med.render("CONFIRMED!", True, CONFIRM_COLOR)
            screen.blit(surf, (8, y))
        y += 22
    else:
        surf = font_sml.render("No gesture", True, (130, 130, 130))
        screen.blit(surf, (8, y))

    # ── Mode banners — stack up from bottom ───────────────────────────────────
    banner_h = 28
    banner_y = OVERLAY_H

    if caps_active:
        banner_y -= banner_h
        pygame.draw.rect(screen, (0, 70, 150),
                         (0, banner_y, OVERLAY_W, banner_h))
        surf = font_med.render("CAPS LOCK ON", True, (255, 255, 255))
        screen.blit(surf, ((OVERLAY_W - surf.get_width()) // 2,
                           banner_y + (banner_h - surf.get_height()) // 2))

    if hold_mode:
        banner_y -= banner_h
        if spam_key_name:
            is_tap = spam_key_name in TAP_REPEAT_KEYS
            label = f"{'tap' if is_tap else 'hold'}  '{spam_key_name}'"
            pygame.draw.rect(screen, (0, 100, 35),
                             (0, banner_y, OVERLAY_W, banner_h))
        else:
            label = "HOLD MODE"
            pygame.draw.rect(screen, (80, 15, 80),
                             (0, banner_y, OVERLAY_W, banner_h))
        surf = font_med.render(label, True, (255, 255, 255))
        screen.blit(surf, ((OVERLAY_W - surf.get_width()) // 2,
                           banner_y + (banner_h - surf.get_height()) // 2))

    pygame.display.flip()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global gesture_buffer, last_confirmed_gesture
    global caps_lock_active, hip_y_history, jump_cooldown
    global hold_mode_active, spam_key, last_spam_time, spam_debounce_counter
    global torso_height_history, bow_cooldown

    parser = argparse.ArgumentParser(
        description="Semaphore-to-keyboard converter")
    parser.add_argument('--type', '-t', action='store_true',
                        help='Actually send keypresses (default: display-only mode)')
    parser.add_argument('--tolerance', type=int, default=SNAP_TOLERANCE,
                        help=f'Angle snap tolerance in degrees (default: {SNAP_TOLERANCE})')
    args = parser.parse_args()

    log_file = None
    if OUTPUT_FILE:
        log_file = open(OUTPUT_FILE, 'a', encoding='utf-8')
        from datetime import datetime
        log_file.write(f"\n=== Session started {datetime.now()} ===\n")
        log_file.flush()
        print(f"Logging confirmed gestures to: {OUTPUT_FILE}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    # ── Pygame transparent overlay ────────────────────────────────────────────
    screen = _make_overlay_window()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_count = 0
        pose_landmarks_cache = None   # last known landmarks for the overlay

        while cap.isOpened():
            # Handle pygame quit / ESC
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    cap.release()
                    pygame.quit()
                    return

            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            detected_gesture = None
            l_ang = r_ang = snapped_l = snapped_r = None

            if result.pose_landmarks:
                pose_landmarks_cache = result.pose_landmarks[0]
                body = [
                    {'x': 1 - lm.x, 'y': 1 - lm.y, 'visibility': lm.visibility}
                    for lm in result.pose_landmarks[0]
                ]

                if detect_jump(body):
                    caps_lock_active = not caps_lock_active
                    state_label = "ON" if caps_lock_active else "OFF"
                    print(f"[{frame_count}] 🔼 JUMP — CAPS LOCK {state_label}")
                    if log_file:
                        from datetime import datetime
                        log_file.write(
                            f"[{datetime.now().strftime('%H:%M:%S')}] [CAPS LOCK {state_label}]\n")
                        log_file.flush()

                if detect_bow(body):
                    hold_mode_active = not hold_mode_active
                    mode_label = "HOLD" if hold_mode_active else "TAP"
                    print(
                        f"[{frame_count}] 🙇 BOW — switched to {mode_label} mode")
                    if log_file:
                        from datetime import datetime
                        log_file.write(
                            f"[{datetime.now().strftime('%H:%M:%S')}] [MODE: {mode_label}]\n")
                        log_file.flush()
                    if not hold_mode_active:
                        release_held_key(args.type)

                armL = (body[11], body[13], body[15])
                armR = (body[12], body[14], body[16])

                if is_limb_pointing(*armL) and is_limb_pointing(*armR):
                    l_ang = get_limb_direction(armL)
                    r_ang = get_limb_direction(armR)
                    snapped_l, snapped_r, match = snap_to_nearest_semaphore(
                        l_ang, r_ang, tolerance=args.tolerance)
                    if match:
                        detected_gesture = match['a']
                        print(f"[{frame_count}] Gesture: '{detected_gesture}' "
                              f"(L={l_ang}°→{snapped_l}°, R={r_ang}°→{snapped_r}°) "
                              f"| Buffer: {len(gesture_buffer)}/{GESTURE_CONFIRMATION_FRAMES}")
                    else:
                        print(f"[{frame_count}] No semaphore match — "
                              f"L={l_ang}°, R={r_ang}° (tolerance={args.tolerance}°)")
                else:
                    l_vis = all(j['visibility'] >=
                                VISIBILITY_THRESHOLD for j in armL)
                    r_vis = all(j['visibility'] >=
                                VISIBILITY_THRESHOLD for j in armR)
                    if l_vis and r_vis:
                        print(f"[{frame_count}] Arms visible but not straight — "
                              f"L_pointing={is_limb_pointing(*armL)}, "
                              f"R_pointing={is_limb_pointing(*armR)}")

            confirmed = check_gesture_confirmation(detected_gesture)

            if hold_mode_active:
                if detected_gesture is not None:
                    spam_debounce_counter = 0
                else:
                    spam_debounce_counter += 1
                    if spam_debounce_counter >= SPAM_DEBOUNCE_FRAMES:
                        if spam_key is not None:
                            release_held_key(args.type)
                        spam_debounce_counter = 0

                if spam_key is not None and spam_key in TAP_REPEAT_KEYS:
                    now = time.time()
                    if now - last_spam_time >= 1.0 / SPAM_RATE:
                        fire_keypress(spam_key, args.type)
                        last_spam_time = now
                        if args.type:
                            print(f"  💥 Tap-repeat: '{spam_key}'")

            if confirmed:
                output_gesture(confirmed, frame, args.type, log_file,
                               caps_lock_active, hold_mode_active)

            draw_overlay(screen, pose_landmarks_cache,
                         detected_gesture, len(gesture_buffer),
                         bool(confirmed), caps_lock_active,
                         hold_mode_active, spam_key)

    if spam_key is not None:
        release_held_key(args.type)

    cap.release()
    pygame.quit()
    if log_file:
        from datetime import datetime
        log_file.write(f"=== Session ended {datetime.now()} ===\n")
        log_file.close()
        print(f"Log saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
