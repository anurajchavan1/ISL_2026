"""
live_inference.py — Integrated ISL Real-Time Translator

Combines:
  1. Static sign model  — recognises hand poses (letters, static signs)
  2. Dynamic sign model — recognises motion-based signs (words)
  3. Groq LLM           — converts recognised word sequences to English sentences

How the two models work together:
  - Every frame is processed by both models simultaneously.
  - The dynamic model runs on a 30-frame rolling buffer (motion-based signs).
  - The static model runs on the current single frame (pose-based signs).
  - The dynamic model takes priority when it is confident.
  - The static model contributes when the dynamic model is uncertain AND
    the hand is not moving (static sign being held).
  - Words from both models feed into the same SentenceBuilder.

Controls:
  T = force sentence completion (sends to Groq)
  C = clear everything
  M = toggle which model is shown in debug overlay
  Q = quit
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from groq import Groq
import datetime
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

# ── Groq ─────────────────────────────────────────────────────────────────────
# GROQ_API_KEY = ""   # ← paste your key from console.groq.com
groq_client  = os.environ.get("GROQ_API_KEY")

# ── Models ────────────────────────────────────────────────────────────────────
# Dynamic model (Bi-LSTM or TCN trained on motion sequences)
DYNAMIC_MODEL_FILE = 'isl_tcn_model.h5'       # change to isl_bilstm_model.h5 if needed
DYNAMIC_CLASSES    = 'classes.npy'

# Static model (your earlier model trained on hand pose images/landmarks)
# Change this path to wherever your static model is saved
STATIC_MODEL_FILE  = 'isl_static_model.h5'
STATIC_CLASSES     = 'static_classes.npy'

# ── Inference thresholds ──────────────────────────────────────────────────────
DYNAMIC_CONFIDENCE_THRESHOLD = 0.60
STATIC_CONFIDENCE_THRESHOLD  = 0.85   # higher threshold for static — it sees every frame
WRIST_MOTION_THRESHOLD       = 0.004
ANGLE_MOTION_THRESHOLD       = 0.002
STABLE_FRAMES_REQUIRED       = 2      # frames dynamic must agree before accepting
STATIC_STABLE_FRAMES         = 8      # frames static must hold same sign before accepting
INFERENCE_EVERY_N            = 4      # run dynamic inference every N frames

# ── Words to suppress ─────────────────────────────────────────────────────────
# Signs that are frequently falsely detected between other signs.
# Add any word your system repeatedly detects incorrectly.
SUPPRESSED_WORDS = {
    'it',       # detected repeatedly due to residual hand position
}

# ── Time-based disambiguation ─────────────────────────────────────────────────
# Used to resolve good_morning vs good_afternoon confusion
MORNING_CUTOFF_HOUR = 12   # before 12pm → prefer good morning

# ── MediaPipe face indices ────────────────────────────────────────────────────
SELECTED_FACE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    46, 53, 52, 65, 55, 285, 295, 282, 283, 276,
    33, 133, 362, 263,
    1, 2, 98, 327, 152, 148
]

mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils


# ==========================================
# 2. LOAD MODELS
# ==========================================
print(f"Loading dynamic model: {DYNAMIC_MODEL_FILE}")
dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_FILE)

try:
    DYNAMIC_ACTIONS = np.load(DYNAMIC_CLASSES)
    print(f"  Dynamic classes: {len(DYNAMIC_ACTIONS)}")
except FileNotFoundError:
    print("  Warning: classes.npy not found.")
    DYNAMIC_ACTIONS = np.array([])

# Static model loading — gracefully handles missing model
static_model   = None
STATIC_ACTIONS = np.array([])

if os.path.exists(STATIC_MODEL_FILE):
    print(f"Loading static model: {STATIC_MODEL_FILE}")
    static_model = tf.keras.models.load_model(STATIC_MODEL_FILE)
    try:
        STATIC_ACTIONS = np.load(STATIC_CLASSES)
        print(f"  Static classes: {len(STATIC_ACTIONS)}")
    except FileNotFoundError:
        print("  Warning: static_classes.npy not found. Static model disabled.")
        static_model = None
else:
    print(f"Static model not found at '{STATIC_MODEL_FILE}' — running dynamic only.")
    print("  To enable: set STATIC_MODEL_FILE to your static model path.")


# ==========================================
# 3. GROQ SENTENCE TRANSLATION
# ==========================================
def words_to_sentence(word_list):
    """
    Sends recognised ISL word sequence to Groq Llama3 and returns
    a natural English sentence.

    Falls back to rule-based grammar if API is unavailable.
    """
    if not word_list:
        return ""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Indian Sign Language (ISL) interpreter. "
                        "ISL follows Subject-Object-Verb (SOV) word order. "
                        "Question signs appear at the end of the sign sequence. "
                        "Convert the given ISL sign sequence into one natural "
                        "English sentence. Output only the sentence — no explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"ISL signs (in order): {', '.join(word_list)}"
                }
            ],
            max_tokens=120,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  [Groq error] {e} — using fallback grammar.")
        return _fallback_grammar(word_list)


def _fallback_grammar(words):
    if not words:
        return ""
    s  = words.copy()
    qw = {"what", "where", "who", "how", "why", "when"}
    if s[-1].lower() in qw:
        s.insert(0, s.pop(-1))
    pronouns = {"i", "you", "he", "she", "we", "they", "it"}
    if len(s) == 3 and s[0].lower() in pronouns:
        s[1], s[2] = s[2], s[1]
    out = " ".join(s).capitalize()
    return out + ("?" if any(w in out.lower() for w in qw) else ".")


# ==========================================
# 4. TIME-BASED DISAMBIGUATION
# ==========================================
def disambiguate(predicted_label):
    """
    Resolves specific sign confusions using context that the model
    cannot access — primarily system clock for time-of-day signs.

    good_morning vs good_afternoon:
      The two signs differ only in the time gesture component.
      Since we know the actual time, we can correct this confusion.

    Add more rules here as you discover new confusions in live testing.
    """
    hour = datetime.datetime.now().hour

    # Normalise label for comparison (strip numbers and dots)
    clean = (''.join([c for c in predicted_label if not c.isdigit()])
             .replace('.', '').strip().lower())

    # Time-of-day disambiguation
    if 'good' in clean and ('morning' in clean or 'afternoon' in clean):
        if hour < MORNING_CUTOFF_HOUR:
            return predicted_label if 'morning' in clean else predicted_label.replace('afternoon', 'morning')
        else:
            return predicted_label if 'afternoon' in clean else predicted_label.replace('morning', 'afternoon')

    return predicted_label


# ==========================================
# 5. FEATURE EXTRACTION (dynamic model)
# ==========================================
def compute_joint_angles(hand_landmarks_flat):
    if np.all(hand_landmarks_flat == 0):
        return np.zeros(15)
    pts = hand_landmarks_flat.reshape(21, 3)
    triplets = [
        (0,1,2),(1,2,3),
        (0,5,6),(5,6,7),(6,7,8),
        (0,9,10),(9,10,11),(10,11,12),
        (0,13,14),(13,14,15),(14,15,16),
        (0,17,18),(17,18,19),(18,19,20),
        (5,9,13),
    ]
    angles = []
    for a, b, c in triplets:
        v1, v2 = pts[a]-pts[b], pts[c]-pts[b]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        angles.append(0.0 if n1 < 1e-6 or n2 < 1e-6
                      else np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1., 1.)))
    return np.array(angles)


def extract_keypoints(results):
    """345-dim nose-normalised coordinate vector for dynamic model."""
    pose = (np.array([[r.x,r.y,r.z] for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(99))
    face = (np.array([[results.face_landmarks.landmark[i].x,
                       results.face_landmarks.landmark[i].y,
                       results.face_landmarks.landmark[i].z]
                      for i in SELECTED_FACE_INDICES]).flatten()
            if results.face_landmarks else np.zeros(120))
    lh = (np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(63))
    rh = (np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(63))
    raw = np.concatenate([pose, face, lh, rh])
    if results.pose_landmarks:
        nx = results.pose_landmarks.landmark[0].x
        ny = results.pose_landmarks.landmark[0].y
        nz = results.pose_landmarks.landmark[0].z
        raw[0::3] -= nx; raw[1::3] -= ny; raw[2::3] -= nz
    return raw


def extract_static_features(results):
    """
    Feature vector for the static model.
    Uses only hand landmarks (no pose or face) since static signs
    are distinguished by hand shape alone.

    IMPORTANT: This must match whatever feature format your static
    model was trained on. If your static model was trained on raw
    images (CNN), pass the frame directly instead. Adjust this
    function to match your static model's input format.

    Default assumption: static model takes flattened hand landmarks
    (same format as most MediaPipe-based static sign classifiers).
    """
    lh = (np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(63))
    rh = (np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(63))

    # Normalise each hand to its wrist position so prediction is
    # position-invariant (same as most static model training pipelines)
    def normalise_hand(hand_flat):
        if np.all(hand_flat == 0):
            return hand_flat
        pts = hand_flat.reshape(21, 3)
        wrist = pts[0].copy()
        pts  -= wrist
        scale = np.max(np.abs(pts))
        if scale > 1e-6:
            pts /= scale
        return pts.flatten()

    lh = normalise_hand(lh)
    rh = normalise_hand(rh)
    return np.concatenate([lh, rh])   # 126-dim


# ==========================================
# 6. MOTION GATE
# ==========================================
def compute_motion_energy(sequence):
    seq     = np.array(sequence)
    wrist_e = np.mean(np.abs(np.diff(seq[:, 219:225], axis=0)))
    angle_e = np.mean(np.abs(np.diff(seq[:, 690:720], axis=0)))
    return wrist_e, angle_e


# ==========================================
# 7. SENTENCE BUILDER
# ==========================================
class SentenceBuilder:
    PAUSE_FRAMES = 25
    MAX_WORDS    = 8

    def __init__(self):
        self.word_buffer      = []
        self.sentence_history = []
        self.no_hand_counter  = 0
        self.last_word        = ""

    def add_word(self, word, source="dynamic"):
        clean = (''.join([c for c in word if not c.isdigit()])
                 .replace('.', '').strip().lower())
        if not clean:
            return
        # Suppress known false-positive words
        if clean in SUPPRESSED_WORDS:
            print(f"  [SUPPRESSED] '{clean}' — in suppression list")
            return
        if clean == self.last_word:
            return
        self.word_buffer.append(clean)
        self.last_word = clean
        print(f"  [{source.upper()}] Word added: '{clean}'  |  Buffer: {self.word_buffer}")
        if len(self.word_buffer) >= self.MAX_WORDS:
            self._complete()

    def tick(self, hands_visible):
        if hands_visible:
            self.no_hand_counter = 0
            return None
        self.no_hand_counter += 1
        if self.no_hand_counter >= self.PAUSE_FRAMES and self.word_buffer:
            return self._complete()
        return None

    def force_complete(self):
        return self._complete() if self.word_buffer else None

    def clear(self):
        self.word_buffer      = []
        self.sentence_history = []
        self.last_word        = ""
        self.no_hand_counter  = 0

    def _complete(self):
        print(f"\n  Sending to Groq: {self.word_buffer}")
        sentence = words_to_sentence(self.word_buffer)
        self.sentence_history.append(sentence)
        if len(self.sentence_history) > 4:
            self.sentence_history = self.sentence_history[-4:]
        print(f"  Words    : {self.word_buffer}")
        print(f"  Sentence : {sentence}\n")
        self.word_buffer     = []
        self.last_word       = ""
        self.no_hand_counter = 0
        return sentence


# ==========================================
# 8. STATE
# ==========================================
sequence           = deque(maxlen=30)
frame_counter      = 0
prev_keypoints     = None

# Dynamic model state
dyn_current_target = ""
dyn_pred_count     = 0
dyn_conf_history   = deque(maxlen=STABLE_FRAMES_REQUIRED)

# Static model state
stat_current_target = ""
stat_hold_count     = 0

builder          = SentenceBuilder()
current_sentence = ""
show_debug       = True


# ==========================================
# 9. MAIN WEBCAM LOOP
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("\nWebcam started.")
print(f"  Dynamic model  : {DYNAMIC_MODEL_FILE} ({len(DYNAMIC_ACTIONS)} classes)")
print(f"  Static model   : {'enabled ('+str(len(STATIC_ACTIONS))+' classes)' if static_model else 'disabled'}")
print(f"  LLM            : Groq / llama3-8b-8192")
print("  T=complete | C=clear | M=toggle debug | Q=quit\n")

with mp_holistic.Holistic(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Uncomment if camera is rotated:
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ── Landmarks ────────────────────────────────────────────────────────
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # ── Feature extraction ───────────────────────────────────────────────
        keypoints = extract_keypoints(results)
        velocity  = np.zeros(345) if prev_keypoints is None else keypoints - prev_keypoints
        prev_keypoints = keypoints.copy()

        lh_flat = (np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten()
                   if results.left_hand_landmarks else np.zeros(63))
        rh_flat = (np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten()
                   if results.right_hand_landmarks else np.zeros(63))

        combined = np.concatenate([
            keypoints, velocity,
            compute_joint_angles(lh_flat),
            compute_joint_angles(rh_flat)
        ])  # 720-dim
        sequence.append(combined)
        frame_counter += 1

        hands_visible  = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        wrist_e, angle_e = compute_motion_energy(sequence) if len(sequence)==30 else (0., 0.)
        is_moving      = (wrist_e > WRIST_MOTION_THRESHOLD and
                          angle_e > ANGLE_MOTION_THRESHOLD)

        # ── Sentence builder pause tick ──────────────────────────────────────
        completed = builder.tick(hands_visible)
        if completed:
            current_sentence = completed

        # ── Dynamic model inference ──────────────────────────────────────────
        dyn_label = ""
        dyn_conf  = 0.0

        if len(sequence) == 30 and frame_counter % INFERENCE_EVERY_N == 0:
            if hands_visible and is_moving:
                inp = np.expand_dims(np.array(sequence), axis=0)
                res = dynamic_model(inp, training=False).numpy()[0]
                best_idx  = np.argmax(res)
                dyn_conf  = res[best_idx]
                dyn_label = disambiguate(DYNAMIC_ACTIONS[best_idx])

                if dyn_conf > DYNAMIC_CONFIDENCE_THRESHOLD:
                    if dyn_label == dyn_current_target:
                        dyn_pred_count += 1
                        dyn_conf_history.append(dyn_conf)
                    else:
                        dyn_current_target = dyn_label
                        dyn_pred_count     = 1
                        dyn_conf_history.clear()
                        dyn_conf_history.append(dyn_conf)

                    if (dyn_pred_count >= STABLE_FRAMES_REQUIRED and
                            np.mean(dyn_conf_history) > DYNAMIC_CONFIDENCE_THRESHOLD):
                        builder.add_word(dyn_current_target, source="dynamic")
                        dyn_pred_count     = 0
                        dyn_conf_history.clear()
                        dyn_current_target = ""
                else:
                    dyn_pred_count     = 0
                    dyn_current_target = ""
                    dyn_conf_history.clear()
            else:
                if not hands_visible:
                    prev_keypoints = None
                dyn_pred_count     = 0
                dyn_current_target = ""
                dyn_conf_history.clear()

        # ── Static model inference ───────────────────────────────────────────
        # Runs when:
        #   1. Static model is loaded
        #   2. Hands are visible
        #   3. Hand is NOT moving (holding a static sign)
        #   4. Dynamic model is not currently in the middle of confirming a sign
        #
        # Priority: dynamic model takes over when is_moving is True.
        stat_label = ""
        stat_conf  = 0.0

        if (static_model is not None
                and hands_visible
                and not is_moving
                and dyn_pred_count == 0):

            static_features = extract_static_features(results)

            # Only run if at least one hand detected (not all zeros)
            if not np.all(static_features == 0):
                stat_inp  = np.expand_dims(static_features, axis=0)
                stat_res  = static_model(stat_inp, training=False).numpy()[0]
                stat_idx  = np.argmax(stat_res)
                stat_conf = stat_res[stat_idx]
                stat_label = STATIC_ACTIONS[stat_idx]

                if stat_conf > STATIC_CONFIDENCE_THRESHOLD:
                    if stat_label == stat_current_target:
                        stat_hold_count += 1
                    else:
                        stat_current_target = stat_label
                        stat_hold_count     = 1

                    # Accept static sign only after holding for STATIC_STABLE_FRAMES
                    if stat_hold_count >= STATIC_STABLE_FRAMES:
                        builder.add_word(stat_current_target, source="static")
                        stat_hold_count     = 0
                        stat_current_target = ""
                else:
                    stat_hold_count     = 0
                    stat_current_target = ""
            else:
                stat_hold_count     = 0
                stat_current_target = ""

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            c = builder.force_complete()
            if c: current_sentence = c
        elif key == ord('c'):
            builder.clear()
            current_sentence = ""
            print("  Cleared.")
        elif key == ord('m'):
            show_debug = not show_debug
        elif key == ord('q'):
            break

        # ==========================================
        # UI OVERLAY
        # ==========================================
        h, w = image.shape[:2]

        # ── Word buffer bar ──────────────────────────────────────────────────
        cv2.rectangle(image, (0,0), (w,42), (45,45,45), -1)
        wtext = ("Words: " + "  |  ".join(builder.word_buffer)
                 if builder.word_buffer else "Words: (waiting...)")
        cv2.putText(image, wtext, (5,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        # ── Sentence history panel ───────────────────────────────────────────
        n_sent  = len(builder.sentence_history)
        panel_h = 26 * (n_sent + 1)
        panel_y = 48
        cv2.rectangle(image, (0,panel_y), (w, panel_y+panel_h), (25,25,25), -1)
        cv2.putText(image, "Sentences:", (5, panel_y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140,140,140), 1, cv2.LINE_AA)
        for i, sent in enumerate(builder.sentence_history):
            brightness = int(140 + 115*((i+1)/max(n_sent,1)))
            cv2.putText(image, sent, (5, panel_y+18+(i+1)*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (brightness,brightness,brightness), 2, cv2.LINE_AA)

        # ── Status bar ───────────────────────────────────────────────────────
        cv2.rectangle(image, (0, h-90), (w, h), (30,30,30), -1)

        sc = (0,220,80) if (is_moving and hands_visible) else (80,140,220)
        st = ("SIGNING"  if (is_moving and hands_visible) else
              "STATIC"   if (hands_visible and not is_moving) else "NO HANDS")
        cv2.putText(image, st, (5, h-65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2, cv2.LINE_AA)

        # Dynamic model prediction
        if dyn_label and show_debug:
            dc = (0,220,80) if dyn_conf > DYNAMIC_CONFIDENCE_THRESHOLD else (0,165,255)
            cv2.putText(image, f"DYN: {dyn_label} {dyn_conf:.2f}",
                        (5, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, dc, 2, cv2.LINE_AA)

        # Static model prediction
        if stat_label and show_debug:
            sc2 = (0,220,80) if stat_conf > STATIC_CONFIDENCE_THRESHOLD else (0,165,255)
            cv2.putText(image, f"STA: {stat_label} {stat_conf:.2f}",
                        (5, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc2, 2, cv2.LINE_AA)

        if show_debug:
            cv2.putText(image, f"wrist:{wrist_e:.4f} angle:{angle_e:.4f}",
                        (200, h-65), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (160,160,160), 1, cv2.LINE_AA)

        # ── Confirmation bar (dynamic, green) ────────────────────────────────
        if dyn_pred_count > 0 and dyn_current_target:
            bw = int((dyn_pred_count / STABLE_FRAMES_REQUIRED) * 180)
            cv2.rectangle(image, (w-192,8),  (w-10,34),     (50,50,50),  -1)
            cv2.rectangle(image, (w-192,8),  (w-192+bw,34), (0,200,100), -1)
            cv2.putText(image, dyn_current_target[:22], (w-190,27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)

        # ── Hold bar (static, purple) ────────────────────────────────────────
        if stat_hold_count > 0 and stat_current_target:
            bw = int((stat_hold_count / STATIC_STABLE_FRAMES) * 180)
            cv2.rectangle(image, (w-192,38), (w-10,54),     (50,50,50),   -1)
            cv2.rectangle(image, (w-192,38), (w-192+bw,54), (180,0,200),  -1)
            cv2.putText(image, f"[S]{stat_current_target[:18]}", (w-190,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)

        # ── Pause bar (blue) ─────────────────────────────────────────────────
        if builder.no_hand_counter > 0 and builder.word_buffer:
            pp = min(builder.no_hand_counter / SentenceBuilder.PAUSE_FRAMES, 1.0)
            bw = int(pp * 182)
            cv2.rectangle(image, (w-192,58), (w-10,72),     (50,50,50),   -1)
            cv2.rectangle(image, (w-192,58), (w-192+bw,72), (0,160,220),  -1)
            cv2.putText(image, "pause...", (w-190,68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1, cv2.LINE_AA)

        # ── Model + LLM indicator ────────────────────────────────────────────
        indicator = f"DYN:{DYNAMIC_MODEL_FILE.split('_')[1][:3].upper()}"
        if static_model:
            indicator += " + STA"
        indicator += " | LLM:Groq"
        cv2.putText(image, indicator, (5, h-88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,200,255), 1, cv2.LINE_AA)

        cv2.imshow('ISL Real-Time Translator', image)

cap.release()
cv2.destroyAllWindows()