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
import cv2
import mediapipe as mp
import numpy as np
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


# ── OVERLAY WINDOW ────────────────────────────────────────────────────────────
# Black canvas (np.zeros) — black background reads as "nothing" against a dark
# game like Minecraft. The Minecraft paper-doll and HUD are drawn on top.
# Press ESC in the overlay window to quit.
OVERLAY_W = 300              # overlay window width  — resize to taste
OVERLAY_H = 500              # overlay window height — resize to taste
OVERLAY_WIN = "Semaphore Overlay"

# ── Minecraft-palette colours (BGR for OpenCV) ────────────────────────────────
MC_SKIN = (120, 160, 198)   # face / hands
MC_HAIR = (30,  60,  90)   # hair
MC_SHIRT = (180, 100,  55)   # torso / upper arms (blue shirt)
MC_SHIRT_D = (140,  75,  40)   # darker shirt outline
MC_TROUSER = (160,  50,  50)   # trousers
MC_TROUSER_D = (120,  35,  35)   # darker trouser outline
MC_BOOT = (20,  50,  80)   # boots
MC_OUTLINE = (30,  30,  30)   # near-black outline

# ── HUD colours (BGR) ─────────────────────────────────────────────────────────
HUD_COLOR = (0, 220, 255)   # yellow
CONFIRM_COLOR = (120, 255,   0)   # green
WARN_COLOR = (60, 100, 255)   # orange


# ── GEOMETRY ──────────────────────────────────────────────────────────────────

def _rotated_rect_pts(cx, cy, w, h, angle_deg):
    """Return the 4 corners of a rotated rectangle as an int32 numpy array."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    hw, hh = w / 2.0, h / 2.0
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rot = np.array([[c, -s], [s, c]])
    rotated = (rot @ corners.T).T + np.array([cx, cy])
    return rotated.astype(np.int32)


def _draw_block(canvas, fill_bgr, outline_bgr, cx, cy, w, h, angle_deg):
    """Draw one filled + outlined rotated rectangle onto canvas."""
    pts = _rotated_rect_pts(cx, cy, w, h, angle_deg).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pts], fill_bgr)
    cv2.polylines(canvas, [pts], isClosed=True, color=outline_bgr, thickness=2)


def _lm(landmarks, idx):
    """Landmark -> (x, y) pixel coords in the overlay window, X mirrored."""
    lm = landmarks[idx]
    return (int((1 - lm.x) * OVERLAY_W), int(lm.y * OVERLAY_H))


def _angle_deg(p1, p2):
    """Clockwise angle from straight-down of the vector p1 -> p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return degrees(np.arctan2(dx, dy))


def _mid(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


# ── MINECRAFT PAPER DOLL ──────────────────────────────────────────────────────

def _draw_minecraft_character(canvas, landmarks):
    """
    Draw a blocky Minecraft-style 2D paper doll driven by live landmarks.
    Body parts are rotated rectangles whose angles come from the real joints.
    Draw order: right side (back) -> torso -> head -> left side (front).
    """
    if not landmarks:
        return

    lsh = _lm(landmarks, 11)
    rsh = _lm(landmarks, 12)
    lel = _lm(landmarks, 13)
    rel = _lm(landmarks, 14)
    lwr = _lm(landmarks, 15)
    rwr = _lm(landmarks, 16)
    lhip = _lm(landmarks, 23)
    rhip = _lm(landmarks, 24)
    lkn = _lm(landmarks, 25)
    rkn = _lm(landmarks, 26)
    lank = _lm(landmarks, 27)
    rank = _lm(landmarks, 28)

    sh_mid = _mid(lsh,  rsh)
    hip_mid = _mid(lhip, rhip)

    # Scale block sizes to shoulder width so they adapt to camera distance
    sw = max(abs(lsh[0] - rsh[0]), 20)
    sc = sw / 60.0

    def blk(w, h):
        return int(w * sc), int(h * sc)

    HW,  HH = blk(50, 50)   # head
    TW,  TH = blk(54, 72)   # torso
    UAW, UAH = blk(20, 38)   # upper arm
    FAW, FAH = blk(18, 36)   # forearm
    TWW, TWH = blk(22, 40)   # thigh
    SHW, SHH = blk(20, 38)   # shin
    BTH = int(12 * sc)  # boot height

    # ── Right leg (back) ─────────────────────────────────────────────────────
    ang = _angle_deg(rhip, rkn)
    _draw_block(canvas, MC_TROUSER, MC_TROUSER_D,
                *_mid(rhip, rkn), TWW, TWH, ang)
    ang = _angle_deg(rkn, rank)
    _draw_block(canvas, MC_TROUSER, MC_TROUSER_D,
                *_mid(rkn, rank), SHW, SHH, ang)
    _draw_block(canvas, MC_BOOT, MC_OUTLINE, *rank, SHW, BTH, ang)

    # ── Right arm (back) ─────────────────────────────────────────────────────
    ang = _angle_deg(rsh, rel)
    _draw_block(canvas, MC_SHIRT, MC_SHIRT_D, *_mid(rsh, rel), UAW, UAH, ang)
    ang = _angle_deg(rel, rwr)
    _draw_block(canvas, MC_SKIN, MC_SHIRT_D, *_mid(rel, rwr), FAW, FAH, ang)

    # ── Torso ─────────────────────────────────────────────────────────────────
    ang = _angle_deg(sh_mid, hip_mid)
    _draw_block(canvas, MC_SHIRT, MC_SHIRT_D, *
                _mid(sh_mid, hip_mid), TW, TH, ang)

    # ── Head ──────────────────────────────────────────────────────────────────
    hx = sh_mid[0]
    hy = sh_mid[1] - int(HH * 0.65)
    _draw_block(canvas, MC_SKIN, MC_OUTLINE, hx, hy, HW, HH, 0)
    hair_h = int(HH * 0.35)
    hair_cy = hy - HH // 2 + hair_h // 2
    _draw_block(canvas, MC_HAIR, MC_OUTLINE, hx, hair_cy, HW, hair_h, 0)
    eye_y = hy - int(HH * 0.08)
    eye_off = int(HW * 0.18)
    eye_sz = max(int(HW * 0.14), 3)
    for ex in [hx - eye_off, hx + eye_off]:
        cv2.rectangle(canvas,
                      (ex - eye_sz // 2, eye_y - eye_sz // 2),
                      (ex + eye_sz // 2, eye_y + eye_sz // 2),
                      (80, 30, 30), -1)

    # ── Left leg (front) ─────────────────────────────────────────────────────
    ang = _angle_deg(lhip, lkn)
    _draw_block(canvas, MC_TROUSER, MC_TROUSER_D,
                *_mid(lhip, lkn), TWW, TWH, ang)
    ang = _angle_deg(lkn, lank)
    _draw_block(canvas, MC_TROUSER, MC_TROUSER_D,
                *_mid(lkn, lank), SHW, SHH, ang)
    _draw_block(canvas, MC_BOOT, MC_OUTLINE, *lank, SHW, BTH, ang)

    # ── Left arm (front) ─────────────────────────────────────────────────────
    ang = _angle_deg(lsh, lel)
    _draw_block(canvas, MC_SHIRT, MC_SHIRT_D, *_mid(lsh, lel), UAW, UAH, ang)
    ang = _angle_deg(lel, lwr)
    _draw_block(canvas, MC_SKIN, MC_SHIRT_D, *_mid(lel, lwr), FAW, FAH, ang)


# ── DRAW OVERLAY ──────────────────────────────────────────────────────────────

def draw_overlay(pose_landmarks, detected_gesture, buffer_count,
                 confirmed, caps_active, hold_mode, spam_key_name):
    """
    Build and show one frame of the black-canvas overlay:
      - Minecraft paper doll driven by live pose landmarks
      - HUD: current letter + confirm progress bar
      - Mode banners stacked at the bottom
    """
    canvas = np.zeros((OVERLAY_H, OVERLAY_W, 3), dtype=np.uint8)

    _draw_minecraft_character(canvas, pose_landmarks)

    # ── HUD — top left ────────────────────────────────────────────────────────
    y = 28
    if detected_gesture:
        display = (detected_gesture.upper()
                   if caps_active and len(detected_gesture) == 1 and detected_gesture.isalpha()
                   else detected_gesture)
        color = CONFIRM_COLOR if (hold_mode and spam_key_name) else HUD_COLOR
        cv2.putText(canvas, display, (12, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 4)
        cv2.putText(canvas, display, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        y += 36

        bar_w = OVERLAY_W - 20
        ratio = min(buffer_count / GESTURE_CONFIRMATION_FRAMES, 1.0)
        done_w = int(bar_w * ratio)
        bar_col = CONFIRM_COLOR if ratio >= 1.0 else WARN_COLOR
        cv2.rectangle(canvas, (10, y), (10 + bar_w, y + 8), (50, 50, 50), -1)
        if done_w > 0:
            cv2.rectangle(canvas, (10, y), (10 + done_w, y + 8), bar_col, -1)
        y += 18

        if confirmed:
            cv2.putText(canvas, "CONFIRMED!", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, CONFIRM_COLOR, 2)
    else:
        cv2.putText(canvas, "No gesture", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

    # ── Mode banners — stack upward from bottom ───────────────────────────────
    BH = 28
    by = OVERLAY_H

    if caps_active:
        by -= BH
        cv2.rectangle(canvas, (0, by), (OVERLAY_W, by + BH), (150, 70, 0), -1)
        tw = cv2.getTextSize(
            "CAPS LOCK ON", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        cv2.putText(canvas, "CAPS LOCK ON",
                    ((OVERLAY_W - tw) // 2, by + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if hold_mode:
        by -= BH
        if spam_key_name:
            is_tap = spam_key_name in TAP_REPEAT_KEYS
            kind = "tap" if is_tap else "hold"
            label = kind + "  '" + spam_key_name + "'"
            cv2.rectangle(canvas, (0, by), (OVERLAY_W,
                          by + BH), (35, 100, 0), -1)
        else:
            label = "HOLD MODE"
            cv2.rectangle(canvas, (0, by), (OVERLAY_W,
                          by + BH), (80, 15, 80), -1)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        cv2.putText(canvas, label,
                    ((OVERLAY_W - tw) // 2, by + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow(OVERLAY_WIN, canvas)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global gesture_buffer, last_confirmed_gesture
    global caps_lock_active, hip_y_history, jump_cooldown
    global hold_mode_active, spam_key, last_spam_time, spam_debounce_counter
    global torso_height_history, bow_cooldown

    parser = argparse.ArgumentParser(
        description="Semaphore-to-keyboard converter")
    parser.add_argument("--type", "-t", action="store_true",
                        help="Actually send keypresses (default: display-only mode)")
    parser.add_argument("--tolerance", type=int, default=SNAP_TOLERANCE,
                        help="Angle snap tolerance in degrees (default: %(default)s)")
    args = parser.parse_args()

    log_file = None
    if OUTPUT_FILE:
        log_file = open(OUTPUT_FILE, "a", encoding="utf-8")
        from datetime import datetime
        log_file.write("\n=== Session started " +
                       str(datetime.now()) + " ===\n")
        log_file.flush()
        print("Logging confirmed gestures to: " + OUTPUT_FILE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Create overlay window; try to make it always-on-top on Windows
    cv2.namedWindow(OVERLAY_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(OVERLAY_WIN, OVERLAY_W, OVERLAY_H)
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, OVERLAY_WIN)
        if hwnd:
            ctypes.windll.user32.SetWindowPos(
                hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    except Exception:
        pass   # Non-Windows: window won't float on top automatically

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_count = 0
        pose_landmarks_cache = None

        while cap.isOpened():
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
                    {"x": 1 - lm.x, "y": 1 - lm.y, "visibility": lm.visibility}
                    for lm in result.pose_landmarks[0]
                ]

                if detect_jump(body):
                    caps_lock_active = not caps_lock_active
                    state_label = "ON" if caps_lock_active else "OFF"
                    print(
                        "[{}] JUMP - CAPS LOCK {}".format(frame_count, state_label))
                    if log_file:
                        from datetime import datetime
                        log_file.write("[{}] [CAPS LOCK {}]\n".format(
                            datetime.now().strftime("%H:%M:%S"), state_label))
                        log_file.flush()

                if detect_bow(body):
                    hold_mode_active = not hold_mode_active
                    mode_label = "HOLD" if hold_mode_active else "TAP"
                    print(
                        "[{}] BOW - switched to {} mode".format(frame_count, mode_label))
                    if log_file:
                        from datetime import datetime
                        log_file.write("[{}] [MODE: {}]\n".format(
                            datetime.now().strftime("%H:%M:%S"), mode_label))
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
                        detected_gesture = match["a"]
                        print("[{}] Gesture: '{}' (L={}->{}  R={}->{})"
                              " | Buffer: {}/{}".format(
                                  frame_count, detected_gesture,
                                  l_ang, snapped_l, r_ang, snapped_r,
                                  len(gesture_buffer), GESTURE_CONFIRMATION_FRAMES))
                    else:
                        print("[{}] No semaphore match - L={}  R={}  (tol={})".format(
                            frame_count, l_ang, r_ang, args.tolerance))
                else:
                    l_vis = all(j["visibility"] >=
                                VISIBILITY_THRESHOLD for j in armL)
                    r_vis = all(j["visibility"] >=
                                VISIBILITY_THRESHOLD for j in armR)
                    if l_vis and r_vis:
                        print("[{}] Arms visible but not straight - L={} R={}".format(
                            frame_count,
                            is_limb_pointing(*armL),
                            is_limb_pointing(*armR)))

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
                            print("  Tap-repeat: '{}'".format(spam_key))

            if confirmed:
                output_gesture(confirmed, None, args.type, log_file,
                               caps_lock_active, hold_mode_active)

            draw_overlay(pose_landmarks_cache,
                         detected_gesture, len(gesture_buffer),
                         bool(confirmed), caps_lock_active,
                         hold_mode_active, spam_key)

            if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
                break

    if spam_key is not None:
        release_held_key(args.type)

    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        from datetime import datetime
        log_file.write("=== Session ended " + str(datetime.now()) + " ===\n")
        log_file.close()
        print("Log saved to: " + OUTPUT_FILE)


if __name__ == "__main__":
    main()
