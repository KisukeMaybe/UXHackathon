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
import json
import socket
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist
from math import atan2, degrees

# ── TUNING THRESHOLDS ─────────────────────────────────────────────────────────
VISIBILITY_THRESHOLD = 0.35   # Min landmark confidence to be considered visible
# Max degrees of bend for a "straight" arm (degrees)
STRAIGHT_LIMB_MARGIN = 35
EXTENDED_LIMB_MARGIN = 0.65   # Forearm must be >= this fraction of upper-arm length
# Max per-arm angular error for semaphore snapping (degrees)
SNAP_TOLERANCE = 15
# Consecutive frames required to confirm a gesture
GESTURE_CONFIRMATION_FRAMES = 5
ANGLE_SNAP_STEP = 45    # Angle grid size — keep at 45 to match semaphore table

# ── JUMP DETECTION ────────────────────────────────────────────────────────────
# Hip rise (normalised 0–1) above baseline to count as a jump
JUMP_THRESHOLD = 0.06
# Frames of hip history used to compute the standing baseline
JUMP_BASELINE_FRAMES = 20
# Frames to suppress further jumps after one fires (~1 s at 30 fps)
JUMP_COOLDOWN_FRAMES = 30

# ── BOW DETECTION ─────────────────────────────────────────────────────────────
# Torso-height shrinkage (normalised 0–1) required to count as a bow
BOW_THRESHOLD = 0.08
BOW_BASELINE_FRAMES = 20     # Frames of torso-height history for the upright baseline
# Frames to suppress re-triggering after a bow fires (~1.3 s at 30 fps)
BOW_COOLDOWN_FRAMES = 40

# ── HOLD / SPAM MODE ──────────────────────────────────────────────────────────
SPAM_RATE = 10    # Key presses per second — only used for keys listed in TAP_REPEAT_KEYS
# Consecutive frames of no gesture before hold/spam stops.
SPAM_DEBOUNCE_FRAMES = 8
# Prevents a single wobbly frame from interrupting the hold.

# Keys that should be tapped repeatedly rather than truly held down.
# Everything else (w, a, s, d, space, arrow keys, etc.) will be held with
# keyboard.press() and released with keyboard.release() for smooth movement.
TAP_REPEAT_KEYS = {"escape", "tab", "enter", "backspace", "capslock"}

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "semaphore_output.txt"   # Set to None to disable file logging

# ── CAMERA / MODEL ────────────────────────────────────────────────────────────
MODEL_PATH = 'pose_landmarker_heavy.task'
CAMERA_INDEX = 0

# ── SKELETON CONNECTIONS (landmark index pairs) ───────────────────────────────
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),   # Left arm
    (12, 14), (14, 16),             # Right arm
    (11, 23), (12, 24), (23, 24),   # Torso
    (23, 25), (25, 27),             # Left leg
    (24, 26), (26, 28),             # Right leg
]

# ── SEMAPHORE LOOKUP TABLE ────────────────────────────────────────────────────
# Keys are (left_arm_angle, right_arm_angle) in degrees, snapped to 45° grid.
# Restricted list: ONLY matches to the closest of these 8 characters.
SEMAPHORES = {
    # MOVEMENT
    (45, 135): {'a': "w"},
    (-45,   225): {'a': "s"},
    (-45,    135): {'a': "a"},
    (45,   225): {'a': "d"},

    # ACTIONS
    #(0,    180): {'a': "e"},
    #(135,    0): {'a': "q"},

    # MENU / OTHER
    #(90,   180): {'a': "escape"},
    #(90,    90): {'a': "space"},    # Space: Both arms straight down
}

# ── GESTURE CONFIRMATION STATE ────────────────────────────────────────────────
gesture_buffer = []    # Rolling list of detected gesture labels
last_confirmed_gesture = None  # Prevents the same gesture firing twice in a row

# ── HOLD MODE / BOW STATE ─────────────────────────────────────────────────────
hold_mode_active = False   # False = tap mode (one press per gesture confirm)
# True  = hold mode (key held down while pose is held)
spam_key = None    # The key currently being held/spammed, or None if idle
last_spam_time = 0.0     # time.time() of last tap-repeat press (TAP_REPEAT_KEYS only)
# Counts consecutive frames with no detected gesture.
spam_debounce_counter = 0
# Only releases the key once this hits SPAM_DEBOUNCE_FRAMES,
# so a single wobbly frame doesn't interrupt a held key.
# Rolling buffer of shoulder-to-hip distances for bow detection
torso_height_history = []
bow_cooldown = 0       # Counts down after a bow fires to prevent re-triggering

# ── JUMP DETECTION STATE ──────────────────────────────────────────────────────
hip_y_history = []            # Rolling buffer of recent hip Y positions (flipped coords)
jump_cooldown = 0             # Counts down after a jump fires to prevent re-triggering

MC_IP = "127.0.0.1"
MC_PORT = 5005
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ── GEOMETRY HELPERS ──────────────────────────────────────────────────────────


def get_angle(a, b, c):
    """Return the angle (degrees) at joint b, formed by points a–b–c."""
    ang = degrees(
        atan2(c['y'] - b['y'], c['x'] - b['x']) -
        atan2(a['y'] - b['y'], a['x'] - b['x'])
    )
    return ang + 360 if ang < 0 else ang


def get_limb_direction(arm):
    """
    Return the arm's direction snapped to the nearest ANGLE_SNAP_STEP grid.
    arm is a 3-tuple: (shoulder, elbow, wrist) landmark dicts.
    Returns an integer angle; 270° is remapped to -90° for consistency.
    """
    dy = arm[2]['y'] - arm[0]['y']
    dx = arm[2]['x'] - arm[0]['x']
    angle = degrees(atan2(dy, dx))

    # Snap to nearest grid step
    mod_close = angle % ANGLE_SNAP_STEP
    angle -= mod_close
    if mod_close > ANGLE_SNAP_STEP / 2:
        angle += ANGLE_SNAP_STEP

    angle = int(angle)
    return -90 if angle == 270 else angle


def is_limb_pointing(upper, mid, lower):
    """
    Return True if the limb (upper→mid→lower) is:
      - all landmarks sufficiently visible,
      - roughly straight (within STRAIGHT_LIMB_MARGIN degrees), and
      - forearm at least EXTENDED_LIMB_MARGIN × upper-arm length.
    """
    if any(j['visibility'] < VISIBILITY_THRESHOLD for j in [upper, mid, lower]):
        return False

    straightness = abs(180 - get_angle(upper, mid, lower))
    if straightness >= STRAIGHT_LIMB_MARGIN:
        return False

    u_len = dist.euclidean([upper['x'], upper['y']], [mid['x'], mid['y']])
    l_len = dist.euclidean([lower['x'], lower['y']], [mid['x'], mid['y']])
    return l_len >= EXTENDED_LIMB_MARGIN * u_len


# ── JUMP DETECTION ────────────────────────────────────────────────────────────

def detect_jump(body):
    """
    Track hip Y over time and return True when the performer jumps.

    Coordinate note: body dicts use flipped Y (1 - lm.y), so:
      y = 1.0  → top of frame (highest position)
      y = 0.0  → bottom of frame (lowest position)
    A jump therefore shows as hip_y INCREASING above the rolling baseline.

    Returns True once per jump (gated by JUMP_COOLDOWN_FRAMES).
    """
    global hip_y_history, jump_cooldown

    # Tick down cooldown every frame regardless
    if jump_cooldown > 0:
        jump_cooldown -= 1

    hipL = body[23]
    hipR = body[24]

    # Only use hip positions when both are confidently visible
    if (hipL['visibility'] < VISIBILITY_THRESHOLD or
            hipR['visibility'] < VISIBILITY_THRESHOLD):
        return False

    current_hip_y = (hipL['y'] + hipR['y']) / 2.0
    hip_y_history.append(current_hip_y)

    # Keep buffer to JUMP_BASELINE_FRAMES length
    if len(hip_y_history) > JUMP_BASELINE_FRAMES:
        hip_y_history.pop(0)

    # Need a full baseline window before we can reliably detect jumps
    if len(hip_y_history) < JUMP_BASELINE_FRAMES:
        return False

    # Baseline = average of the older frames (exclude the newest 3 which may
    # already be mid-jump)
    baseline = sum(hip_y_history[:-3]) / len(hip_y_history[:-3])

    if current_hip_y > baseline + JUMP_THRESHOLD and jump_cooldown == 0:
        jump_cooldown = JUMP_COOLDOWN_FRAMES
        return True

    return False


# ── BOW DETECTION ─────────────────────────────────────────────────────────────

def detect_bow(body):
    """
    Detect a bow pose by tracking the vertical distance between shoulders and hips.

    Coordinate note: body dicts use flipped Y (1 - lm.y), so:
      y = 1.0  → top of frame    y = 0.0  → bottom of frame
    When standing upright, shoulders sit well above hips:
      torso_height = avg_shoulder_y - avg_hip_y  →  positive value
    When bowing forward, shoulders drop toward hips:
      torso_height SHRINKS below the rolling baseline.

    Returns True once per bow (gated by BOW_COOLDOWN_FRAMES).
    """
    global torso_height_history, bow_cooldown

    if bow_cooldown > 0:
        bow_cooldown -= 1

    shoulderL, shoulderR = body[11], body[12]
    hipL,      hipR = body[23], body[24]

    landmarks = [shoulderL, shoulderR, hipL, hipR]
    if any(lm['visibility'] < VISIBILITY_THRESHOLD for lm in landmarks):
        return False

    avg_shoulder_y = (shoulderL['y'] + shoulderR['y']) / 2.0
    avg_hip_y = (hipL['y'] + hipR['y']) / 2.0
    current_torso_height = avg_shoulder_y - avg_hip_y

    torso_height_history.append(current_torso_height)
    if len(torso_height_history) > BOW_BASELINE_FRAMES:
        torso_height_history.pop(0)

    if len(torso_height_history) < BOW_BASELINE_FRAMES:
        return False

    # Baseline from older frames; exclude newest 3 which may already be mid-bow
    baseline = sum(torso_height_history[:-3]) / len(torso_height_history[:-3])

    if current_torso_height < baseline - BOW_THRESHOLD and bow_cooldown == 0:
        bow_cooldown = BOW_COOLDOWN_FRAMES
        return True

    return False


def snap_to_nearest_semaphore(detected_l_ang, detected_r_ang, tolerance=None):
    """
    Find the closest entry in SEMAPHORES to (detected_l_ang, detected_r_ang).
    This version ignores SNAP_TOLERANCE and will ALWAYS "round" to the closest 
    matching character from our restricted dictionary.
    """
    best_match = None
    best_distance = float('inf')

    for (l_ang, r_ang), semaphore in SEMAPHORES.items():
        # Angular distance with wrap-around handling
        l_diff = min(abs(detected_l_ang - l_ang),
                     360 - abs(detected_l_ang - l_ang))
        r_diff = min(abs(detected_r_ang - r_ang),
                     360 - abs(detected_r_ang - r_ang))
        total = l_diff + r_diff

        # Unconditionally accept it if it's the closest one we've checked so far
        if total < best_distance:
            best_distance = total
            best_match = (l_ang, r_ang)

    if best_match:
        return best_match[0], best_match[1], SEMAPHORES[best_match]

    return None, None, None


# ── GESTURE CONFIRMATION ──────────────────────────────────────────────────────

def check_gesture_confirmation(detected_gesture):
    """
    Accumulate detected_gesture into gesture_buffer.
    Returns the gesture label once GESTURE_CONFIRMATION_FRAMES consecutive
    identical frames are seen (and only when it differs from the last output).
    Returns None otherwise.
    """
    global gesture_buffer, last_confirmed_gesture

    if detected_gesture is None:
        gesture_buffer = []
        return None

    # Reset buffer on gesture change
    if gesture_buffer and gesture_buffer[-1] != detected_gesture:
        gesture_buffer = []

    gesture_buffer.append(detected_gesture)

    if len(gesture_buffer) >= GESTURE_CONFIRMATION_FRAMES:
        if detected_gesture != last_confirmed_gesture:
            last_confirmed_gesture = detected_gesture
            gesture_buffer = []
            return detected_gesture

    return None


# ── KEY PRESS / HOLD / RELEASE HELPERS ───────────────────────────────────────

def _resolve_key(key, caps_active):
    """Return the (keyboard_arg, display_string) pair for a key + caps state."""
    return key, key


def _press_key(kb_key, send_keypress):
    """Hold a key down (keyboard.press). Used for movement / sustained input."""
    if send_keypress:
        import keyboard
        keyboard.press(kb_key)


def _release_key(kb_key, send_keypress):
    """Release a previously held key (keyboard.release)."""
    if send_keypress:
        import keyboard
        keyboard.release(kb_key)


def fire_keypress(kb_key, send_keypress):
    """
    Send a single press_and_release for kb_key.
    Used by tap mode and for keys listed in TAP_REPEAT_KEYS in hold mode.
    """
    if send_keypress:
        import keyboard
        keyboard.press_and_release(kb_key)


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def output_gesture(key, frame, send_keypress, log_file, hold_mode):
    """
    Handle a newly confirmed gesture.

    TAP MODE  (hold_mode=False):
        Fire a single press_and_release right now. Done.

    HOLD MODE (hold_mode=True):
        For keys in TAP_REPEAT_KEYS  → store in spam_key for rapid tap-repeat.
        For all other keys (w/a/s/d, space, etc.) → call keyboard.press() to
        physically hold the key down. The main loop calls keyboard.release()
        once the pose is lost. This is what Minecraft needs for smooth movement.
    """
    global spam_key, last_spam_time
    from datetime import datetime

    kb_key, display_key = _resolve_key(key, False)
    timestamp = datetime.now().strftime("%H:%M:%S")
    mode_tag = "[HOLD]" if hold_mode else "[TAP] "
    print(f"[{timestamp}] ✓ {mode_tag} OUTPUT: '{display_key}'")

    if log_file is not None:
        log_file.write(f"[{timestamp}] {mode_tag} {display_key}\n")
        log_file.flush()

    if hold_mode:
        if kb_key in TAP_REPEAT_KEYS:
            # Tap-repeat mode for special keys: store and let the spam ticker fire
            if spam_key != kb_key:
                # Release any previously held key before switching
                if spam_key is not None:
                    _release_key(spam_key, send_keypress)
                    print(
                        f"  → Released '{spam_key}' (switching to tap-repeat '{kb_key}')")
                spam_key = kb_key
                last_spam_time = 0.0   # Fire immediately on the very next tick
                print(f"  → Tap-repeat key set to '{kb_key}'")
        else:
            # True hold: physically press the key down and keep it held.
            # If the key has changed, release the old one first.
            if spam_key != kb_key:
                if spam_key is not None:
                    _release_key(spam_key, send_keypress)
                    print(f"  → Released '{spam_key}'")
                spam_key = kb_key
                _press_key(kb_key, send_keypress)
                print(f"  → Holding '{kb_key}' down")
            # If it's the same key already held, do nothing — it's already down.
    else:
        # Tap mode: one clean press_and_release, nothing stored
        fire_keypress(kb_key, send_keypress)
        if not send_keypress:
            # Display-only: draw the key on the frame as visual feedback
            cv2.putText(frame, display_key, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)


def release_held_key(send_keypress):
    """
    Release whatever key is currently held (spam_key) and clear state.
    Called when the pose is lost in hold mode, or when switching back to tap mode.
    """
    global spam_key, spam_debounce_counter, last_confirmed_gesture
    if spam_key is not None:
        if spam_key not in TAP_REPEAT_KEYS:
            # Only truly held keys need an explicit release
            _release_key(spam_key, send_keypress)
            print(f"  ✋ Released held key '{spam_key}'")
        else:
            print(f"  ✋ Stopped tap-repeat of '{spam_key}'")
        spam_key = None
    spam_debounce_counter = 0
    # Reset so the same gesture can re-trigger immediately after re-posing
    last_confirmed_gesture = None


def send_to_minecraft(landmarks, gesture, is_holding):
    """
    Formats 17 specific landmarks + gesture state into JSON and sends via UDP.
    """
    # 1. Map MediaPipe indices to the 17 nodes (Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles)
    target_indices = [0, 2, 5, 7, 8, 11, 12,
                      13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    nodes = []
    for idx in target_indices:
        lm = landmarks[idx]
        # Use 1920x1080 scaling to match the mock server format
        nodes.append([lm.x * 1920, lm.y * 1080])

    # 2. Build the full payload
    payload = {
        "nodes": nodes,
        "key": str(gesture) if gesture else "None",
        "holding": bool(is_holding),
    }

    # 3. Ship it
    try:
        udp_sock.sendto(json.dumps(payload).encode("utf-8"), (MC_IP, MC_PORT))
    except Exception as e:
        print(f"Socket error: {e}")


# ── DRAWING ───────────────────────────────────────────────────────────────────

def draw_skeleton(image, pose_landmarks):
    """Draw landmark dots and bone lines onto image (in-place)."""
    h, w, _ = image.shape

    for lm in pose_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        s = pose_landmarks[start_idx]
        e = pose_landmarks[end_idx]
        cv2.line(image,
                 (int(s.x * w), int(s.y * h)),
                 (int(e.x * w), int(e.y * h)),
                 (255, 255, 255), 2)


def draw_hud(image, detected_gesture, l_ang, r_ang,
             snapped_l, snapped_r, buffer_count, confirmed,
             hold_mode, spam_key_name):
    """Overlay gesture detection info on the camera frame."""
    h, w, _ = image.shape

    # ── Top info panel ────────────────────────────────────────────────────────
    cv2.rectangle(image, (10, 10), (w - 10, 215), (0, 0, 0), -1)
    cv2.rectangle(image, (10, 10), (w - 10, 215), (255, 255, 255), 2)

    y = 50
    if detected_gesture:
        display = detected_gesture
        # Yellow-orange tint when actively holding/spamming, normal green otherwise
        letter_color = (0, 220, 255) if (
            hold_mode and spam_key_name) else (0, 255, 0)
        cv2.putText(image, f"Letter: {display}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, letter_color, 3)
        y += 55
        cv2.putText(image, f"Raw:     L={l_ang}°  R={r_ang}°", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        y += 35
        cv2.putText(image, f"Snapped: L={snapped_l}°  R={snapped_r}°", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        y += 35
        bar_color = (0, 255, 0) if buffer_count >= GESTURE_CONFIRMATION_FRAMES else (
            0, 165, 255)
        cv2.putText(image, f"Confirm: {buffer_count}/{GESTURE_CONFIRMATION_FRAMES}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
        if confirmed:
            cv2.putText(image, "CONFIRMED!", (w - 280, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        cv2.putText(image, "No gesture detected", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    # ── Banners stack upward from the bottom ──────────────────────────────────
    banner_h = 65
    banner_y = h

    # ── Hold mode banner — only shown when hold_mode is active ───────────────
    if hold_mode:
        banner_y -= banner_h
        if spam_key_name:
            is_tap_repeat = spam_key_name in TAP_REPEAT_KEYS
            label = f"HOLD MODE  —  tapping '{spam_key_name}'" if is_tap_repeat \
                else f"HOLD MODE  —  holding '{spam_key_name}'"
            # Bright green = actively holding/tapping
            cv2.rectangle(image, (0, banner_y),
                          (w, banner_y + banner_h), (0, 170, 60), -1)
            cv2.rectangle(image, (0, banner_y),
                          (w, banner_y + banner_h), (0, 90, 30), 3)
            _draw_banner_text(image, label, banner_y, banner_h, w)
        else:
            # In hold mode but no pose confirmed yet — purple idle
            cv2.rectangle(image, (0, banner_y),
                          (w, banner_y + banner_h), (140, 40, 140), -1)
            cv2.rectangle(image, (0, banner_y),
                          (w, banner_y + banner_h), (80, 0, 80), 3)
            _draw_banner_text(image, "HOLD MODE ON", banner_y, banner_h, w)


def _draw_banner_text(image, label, banner_y, banner_h, w):
    """Draw centred shadowed text inside a banner rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thickness = 3
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = max(10, (w - text_w) // 2)
    text_y = banner_y + (banner_h + text_h) // 2
    cv2.putText(image, label, (text_x + 2, text_y + 2),
                font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, label, (text_x,     text_y),     font,
                font_scale, (255, 255, 255), thickness)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global gesture_buffer, last_confirmed_gesture
    global hold_mode_active, hip_y_history, jump_cooldown
    global hold_mode_active, spam_key, last_spam_time, spam_debounce_counter
    global torso_height_history, bow_cooldown

    parser = argparse.ArgumentParser(
        description="Semaphore-to-keyboard converter")
    parser.add_argument('--type', '-t', action='store_true',
                        help='Actually send keypresses (default: display-only mode)')
    parser.add_argument('--tolerance', type=int, default=SNAP_TOLERANCE,
                        help=f'Angle snap tolerance in degrees (default: {SNAP_TOLERANCE})')
    args = parser.parse_args()

    # Open log file if OUTPUT_FILE is set
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

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_count = 0

        while cap.isOpened():
            cv2.namedWindow('Semaphore', cv2.WINDOW_NORMAL)
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

                body = [
                    {'x': 1 - lm.x, 'y': 1 - lm.y, 'visibility': lm.visibility}
                    for lm in result.pose_landmarks[0]
                ]

                # ── Bow → Hold/Tap mode toggle ─────────────────────────────
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
                    # When switching back to tap mode, release any held key immediately
                    if not hold_mode_active:
                        release_held_key(args.type)

                # ── Arm semaphore detection ────────────────────────────────
                armL = (body[11], body[13], body[15])
                armR = (body[12], body[14], body[16])

                if is_limb_pointing(*armL) and is_limb_pointing(*armR):
                    l_ang = get_limb_direction(armL)
                    r_ang = get_limb_direction(armR)

                    snapped_l, snapped_r, match = snap_to_nearest_semaphore(
                        l_ang, r_ang, tolerance=args.tolerance
                    )

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
                        l_pointing = is_limb_pointing(*armL)
                        r_pointing = is_limb_pointing(*armR)
                        print(f"[{frame_count}] Arms visible but not straight — "
                              f"L_pointing={l_pointing}, R_pointing={r_pointing}")

                send_to_minecraft(
                    result.pose_landmarks[0],
                    detected_gesture,
                    hold_mode_active
                )

                draw_skeleton(frame, result.pose_landmarks[0])
            # ─────────────────────────────────────────────────────────────────
            # HOLD MODE LOGIC
            # ─────────────────────────────────────────────────────────────────
            # This block runs every frame and has two jobs:
            #
            # JOB 1 — Decide whether to keep holding or stop.
            #   When a gesture is detected we reset the debounce counter so
            #   the hold keeps going. When the gesture disappears we increment
            #   the counter. Only once it hits SPAM_DEBOUNCE_FRAMES do we
            #   actually release the key and stop. This stops a single wobbly
            #   camera frame from interrupting a sustained hold pose.
            #
            # JOB 2 — For TAP_REPEAT_KEYS only: fire the key at SPAM_RATE/s.
            #   Regular held keys (w/a/s/d etc.) are already physically held
            #   down by output_gesture — no per-frame action needed for them.
            # ─────────────────────────────────────────────────────────────────

            confirmed = check_gesture_confirmation(detected_gesture)

            if hold_mode_active:

                # ── JOB 1: Release key if pose has been absent long enough ───
                if detected_gesture is not None:
                    # Pose visible this frame — reset debounce so hold continues
                    spam_debounce_counter = 0
                else:
                    spam_debounce_counter += 1
                    if spam_debounce_counter >= SPAM_DEBOUNCE_FRAMES:
                        # Pose truly gone — release the key
                        if spam_key is not None:
                            release_held_key(args.type)
                        spam_debounce_counter = 0

                # ── JOB 2: Tap-repeat for TAP_REPEAT_KEYS ────────────────────
                if spam_key is not None and spam_key in TAP_REPEAT_KEYS:
                    now = time.time()
                    spam_interval = 1.0 / SPAM_RATE

                    if now - last_spam_time >= spam_interval:
                        fire_keypress(spam_key, args.type)
                        last_spam_time = now
                        if args.type:
                            print(f"  💥 Tap-repeat: '{spam_key}'")

            else:
                # ── TAP MODE: nothing extra — output_gesture handles the press.
                pass

            if confirmed:
                output_gesture(confirmed, frame, args.type, log_file,
                               hold_mode_active)

            draw_hud(frame, detected_gesture,
                     l_ang, r_ang, snapped_l, snapped_r,
                     len(gesture_buffer), bool(confirmed),
                     hold_mode_active, spam_key)

            cv2.imshow('Semaphore', frame)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
                break

    # ── Cleanup: make sure no key is left physically held ─────────────────────
    if spam_key is not None:
        release_held_key(args.type)

    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        from datetime import datetime
        log_file.write(f"=== Session ended {datetime.now()} ===\n")
        log_file.close()
        print(f"Log saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()