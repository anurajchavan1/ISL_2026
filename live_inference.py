# """
# live_inference.py

# Real-time ISL translator:
#   - Dynamic model (TCN or Bi-LSTM) for motion-based sign recognition
#   - Static model (optional) for static/fingerspelling signs
#   - Two-part motion gate to suppress false detections between signs
#   - Groq LLM (Llama 3.3 70B) for ISL→English sentence construction
#   - SentenceBuilder with pause-based boundary detection

# Controls:
#   T — force sentence completion → Groq translation
#   C — clear word buffer and sentence history
#   Q — quit
# """

# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from collections import deque
# from groq import Groq
# import datetime
# import os

# # ==========================================
# # CONFIGURATION
# # ==========================================

# # ── API ───────────────────────────────────────────────────────────────────────
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # console.groq.com → free

# # ── Models ────────────────────────────────────────────────────────────────────
# # Use whichever scored higher in diagnose_confusion.py
# DYNAMIC_MODEL_FILE = 'isl_tcn_model.h5'      # or isl_bilstm_model.h5
# DYNAMIC_CLASSES    = 'classes.npy'

# # Static model — set path if you have one, otherwise leave as is
# STATIC_MODEL_FILE  = 'isl_static_model.h5'
# STATIC_CLASSES     = 'static_classes.npy'

# # ── Thresholds (tune these for your environment) ──────────────────────────────
# DYNAMIC_CONF_THRESHOLD = 0.60   # raise to 0.70 if false positives
# STATIC_CONF_THRESHOLD  = 0.85
# WRIST_MOTION_THRESHOLD = 0.004  # raise if signs detected when hands are still
# ANGLE_MOTION_THRESHOLD = 0.002  # raise if signs detected between signs
# STABLE_FRAMES          = 2      # consecutive agreements before word accepted
# STATIC_STABLE_FRAMES   = 8      # frames static sign must be held
# INFERENCE_EVERY_N      = 4      # run dynamic model every N frames

# # ── Sentence settings ─────────────────────────────────────────────────────────
# PAUSE_FRAMES = 25   # no-hand frames before sentence completes
# MAX_WORDS    = 8

# # ── Suppressed words ─────────────────────────────────────────────────────────
# # Words that are frequently false-positived between signs
# # Add any word you observe being wrongly detected repeatedly
# SUPPRESSED_WORDS = {'it'}

# # ── Time disambiguation ───────────────────────────────────────────────────────
# MORNING_CUTOFF = 12  # before noon → prefer good morning over good afternoon

# # ── MediaPipe ─────────────────────────────────────────────────────────────────
# SELECTED_FACE_INDICES = [
#     61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0,
#     267, 269, 270, 409, 46, 53, 52, 65, 55, 285, 295, 282, 283, 276,
#     33, 133, 362, 263, 1, 2, 98, 327, 152, 148
# ]
# mp_holistic = mp.solutions.holistic
# mp_drawing  = mp.solutions.drawing_utils


# # ==========================================
# # LOAD MODELS
# # ==========================================
# groq_client = Groq(api_key=GROQ_API_KEY)

# print(f"Loading dynamic model: {DYNAMIC_MODEL_FILE}")
# dynamic_model   = tf.keras.models.load_model(DYNAMIC_MODEL_FILE)
# DYNAMIC_ACTIONS = np.load(DYNAMIC_CLASSES)
# print(f"  {len(DYNAMIC_ACTIONS)} dynamic classes")

# static_model   = None
# STATIC_ACTIONS = np.array([])
# if os.path.exists(STATIC_MODEL_FILE):
#     print(f"Loading static model: {STATIC_MODEL_FILE}")
#     static_model   = tf.keras.models.load_model(STATIC_MODEL_FILE)
#     if os.path.exists(STATIC_CLASSES):
#         STATIC_ACTIONS = np.load(STATIC_CLASSES)
#         print(f"  {len(STATIC_ACTIONS)} static classes")
#     else:
#         static_model = None
#         print("  static_classes.npy not found — static model disabled")
# else:
#     print("No static model found — running dynamic only")


# # ==========================================
# # GROQ SENTENCE TRANSLATION
# # ==========================================
# def words_to_sentence(word_list):
#     """Converts ISL word sequence to English via Groq Llama 3.3 70B."""
#     if not word_list:
#         return ""
#     try:
#         resp = groq_client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are an Indian Sign Language (ISL) interpreter. "
#                         "ISL follows Subject-Object-Verb (SOV) word order. "
#                         "Question signs appear at the end of the sign sequence. "
#                         "Convert the ISL sign word sequence into one natural "
#                         "English sentence. Output only the sentence — nothing else."
#                     )
#                 },
#                 {
#                     "role": "user",
#                     "content": f"ISL signs in order: {', '.join(word_list)}"
#                 }
#             ],
#             max_tokens=120,
#             temperature=0.3,
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"  [Groq error] {e}")
#         return _fallback(word_list)


# def _fallback(words):
#     """Rule-based fallback when Groq is unavailable."""
#     if not words: return ""
#     s  = words.copy()
#     qw = {"what","where","who","how","why","when"}
#     if s[-1].lower() in qw:
#         s.insert(0, s.pop(-1))
#     pr = {"i","you","he","she","we","they"}
#     if len(s)==3 and s[0].lower() in pr:
#         s[1], s[2] = s[2], s[1]
#     out = " ".join(s).capitalize()
#     return out + ("?" if any(w in out.lower() for w in qw) else ".")


# # ==========================================
# # FEATURE EXTRACTION (must match extract_features.py exactly)
# # ==========================================
# def compute_joint_angles(hand_flat):
#     if np.all(hand_flat == 0): return np.zeros(15)
#     pts = hand_flat.reshape(21,3)
#     triplets = [
#         (0,1,2),(1,2,3),
#         (0,5,6),(5,6,7),(6,7,8),
#         (0,9,10),(9,10,11),(10,11,12),
#         (0,13,14),(13,14,15),(14,15,16),
#         (0,17,18),(17,18,19),(18,19,20),
#         (5,9,13),
#     ]
#     out = []
#     for a,b,c in triplets:
#         v1,v2 = pts[a]-pts[b], pts[c]-pts[b]
#         n1,n2 = np.linalg.norm(v1), np.linalg.norm(v2)
#         out.append(0. if n1<1e-6 or n2<1e-6
#                    else float(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1.,1.))))
#     return np.array(out)


# def extract_keypoints(results):
#     pose = (np.array([[r.x,r.y,r.z]
#                       for r in results.pose_landmarks.landmark]).flatten()
#             if results.pose_landmarks else np.zeros(99))
#     face = (np.array([[results.face_landmarks.landmark[i].x,
#                        results.face_landmarks.landmark[i].y,
#                        results.face_landmarks.landmark[i].z]
#                       for i in SELECTED_FACE_INDICES]).flatten()
#             if results.face_landmarks else np.zeros(120))
#     lh = (np.array([[r.x,r.y,r.z]
#                     for r in results.left_hand_landmarks.landmark]).flatten()
#           if results.left_hand_landmarks else np.zeros(63))
#     rh = (np.array([[r.x,r.y,r.z]
#                     for r in results.right_hand_landmarks.landmark]).flatten()
#           if results.right_hand_landmarks else np.zeros(63))
#     raw = np.concatenate([pose, face, lh, rh])
#     if results.pose_landmarks:
#         nx = results.pose_landmarks.landmark[0].x
#         ny = results.pose_landmarks.landmark[0].y
#         nz = results.pose_landmarks.landmark[0].z
#         raw[0::3]-=nx; raw[1::3]-=ny; raw[2::3]-=nz
#     return raw


# def extract_static_features(results):
#     """Wrist-normalised hand landmarks for static model."""
#     def norm(h):
#         if np.all(h==0): return h
#         pts = h.reshape(21,3)
#         pts -= pts[0]
#         s = np.max(np.abs(pts))
#         if s > 1e-6: pts /= s
#         return pts.flatten()
#     lh = (np.array([[r.x,r.y,r.z]
#                     for r in results.left_hand_landmarks.landmark]).flatten()
#           if results.left_hand_landmarks else np.zeros(63))
#     rh = (np.array([[r.x,r.y,r.z]
#                     for r in results.right_hand_landmarks.landmark]).flatten()
#           if results.right_hand_landmarks else np.zeros(63))
#     return np.concatenate([norm(lh), norm(rh)])  # 126-dim


# # ==========================================
# # MOTION GATE
# # ==========================================
# def motion_energy(sequence):
#     seq     = np.array(sequence)
#     wrist_e = np.mean(np.abs(np.diff(seq[:,219:225], axis=0)))
#     angle_e = np.mean(np.abs(np.diff(seq[:,690:720], axis=0)))
#     return wrist_e, angle_e


# # ==========================================
# # TIME-BASED DISAMBIGUATION
# # ==========================================
# def disambiguate(label):
#     """
#     Corrects time-of-day sign confusions using the system clock.
#     Good Morning vs Good Afternoon is the most common in this dataset.
#     """
#     hour  = datetime.datetime.now().hour
#     clean = label.lower()
#     if 'good' in clean:
#         if 'morning' in clean or 'afternoon' in clean:
#             if hour < MORNING_CUTOFF:
#                 return label if 'morning' in clean else label.replace('afternoon','morning').replace('Afternoon','Morning')
#             else:
#                 return label if 'afternoon' in clean else label.replace('morning','afternoon').replace('Morning','Afternoon')
#     return label


# # ==========================================
# # SENTENCE BUILDER
# # ==========================================
# class SentenceBuilder:
#     def __init__(self):
#         self.words    = []
#         self.history  = []
#         self.no_hand  = 0
#         self.last     = ""

#     def add(self, word, src="dyn"):
#         clean = (''.join(c for c in word if not c.isdigit())
#                  .replace('.','').strip().lower())
#         if not clean or clean in SUPPRESSED_WORDS or clean == self.last:
#             if clean in SUPPRESSED_WORDS:
#                 print(f"  [SUPPRESSED] '{clean}'")
#             return
#         self.words.append(clean)
#         self.last = clean
#         print(f"  [{src.upper()}] '{clean}' → {self.words}")
#         if len(self.words) >= MAX_WORDS:
#             self._done()

#     def tick(self, hands):
#         if hands:
#             self.no_hand = 0
#             return None
#         self.no_hand += 1
#         if self.no_hand >= PAUSE_FRAMES and self.words:
#             return self._done()
#         return None

#     def force(self):
#         return self._done() if self.words else None

#     def clear(self):
#         self.words=[]; self.history=[]; self.no_hand=0; self.last=""

#     def _done(self):
#         print(f"\n  → Groq: {self.words}")
#         s = words_to_sentence(self.words)
#         self.history.append(s)
#         if len(self.history)>4: self.history=self.history[-4:]
#         print(f"  ✓ {s}\n")
#         self.words=[]; self.last=""; self.no_hand=0
#         return s


# # ==========================================
# # STATE
# # ==========================================
# sequence  = deque(maxlen=30)
# prev_kp   = None
# fc        = 0

# dyn_target = ""; dyn_count = 0
# dyn_hist   = deque(maxlen=STABLE_FRAMES)

# stat_target = ""; stat_count = 0

# builder  = SentenceBuilder()
# cur_sent = ""


# # ==========================================
# # MAIN LOOP
# # ==========================================
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("ERROR: Cannot open webcam."); exit()

# print(f"\nStarted | Dynamic: {DYNAMIC_MODEL_FILE} | "
#       f"Static: {'on' if static_model else 'off'} | LLM: Groq")
# print("T=translate  C=clear  Q=quit\n")

# with mp_holistic.Holistic(min_detection_confidence=0.5,
#                            min_tracking_confidence=0.5,
#                            model_complexity=1) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break

#         # Uncomment if camera is rotated:
#         # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img.flags.writeable = False
#         res = holistic.process(img)
#         img.flags.writeable = True
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#         # Landmarks
#         if res.face_landmarks:
#             mp_drawing.draw_landmarks(img, res.face_landmarks,
#                 mp_holistic.FACEMESH_TESSELATION,
#                 mp_drawing.DrawingSpec(color=(80,110,10),  thickness=1, circle_radius=1),
#                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
#         if res.pose_landmarks:
#             mp_drawing.draw_landmarks(img, res.pose_landmarks,
#                 mp_holistic.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
#                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
#         if res.left_hand_landmarks:
#             mp_drawing.draw_landmarks(img, res.left_hand_landmarks,
#                 mp_holistic.HAND_CONNECTIONS)
#         if res.right_hand_landmarks:
#             mp_drawing.draw_landmarks(img, res.right_hand_landmarks,
#                 mp_holistic.HAND_CONNECTIONS)

#         # Feature extraction
#         kp  = extract_keypoints(res)
#         vel = np.zeros(345) if prev_kp is None else kp - prev_kp
#         prev_kp = kp.copy()

#         lh_flat = (np.array([[r.x,r.y,r.z]
#                               for r in res.left_hand_landmarks.landmark]).flatten()
#                    if res.left_hand_landmarks else np.zeros(63))
#         rh_flat = (np.array([[r.x,r.y,r.z]
#                               for r in res.right_hand_landmarks.landmark]).flatten()
#                    if res.right_hand_landmarks else np.zeros(63))

#         combined = np.concatenate([kp, vel,
#                                    compute_joint_angles(lh_flat),
#                                    compute_joint_angles(rh_flat)])  # 720
#         sequence.append(combined)
#         fc += 1

#         hands    = bool(res.left_hand_landmarks or res.right_hand_landmarks)
#         we, ae   = motion_energy(sequence) if len(sequence)==30 else (0.,0.)
#         moving   = we > WRIST_MOTION_THRESHOLD and ae > ANGLE_MOTION_THRESHOLD

#         completed = builder.tick(hands)
#         if completed: cur_sent = completed

#         # ── Dynamic inference ────────────────────────────────────────────────
#         dyn_lbl = ""; dyn_cf = 0.
#         if len(sequence)==30 and fc % INFERENCE_EVERY_N == 0:
#             if hands and moving:
#                 inp = np.expand_dims(np.array(sequence), 0)
#                 out = dynamic_model(inp, training=False).numpy()[0]
#                 bi  = np.argmax(out)
#                 dyn_cf  = out[bi]
#                 dyn_lbl = disambiguate(DYNAMIC_ACTIONS[bi])

#                 if dyn_cf > DYNAMIC_CONF_THRESHOLD:
#                     if dyn_lbl == dyn_target:
#                         dyn_count += 1
#                         dyn_hist.append(dyn_cf)
#                     else:
#                         dyn_target = dyn_lbl
#                         dyn_count  = 1
#                         dyn_hist.clear()
#                         dyn_hist.append(dyn_cf)

#                     if (dyn_count >= STABLE_FRAMES and
#                             np.mean(dyn_hist) > DYNAMIC_CONF_THRESHOLD):
#                         builder.add(dyn_target, "dyn")
#                         dyn_count=0; dyn_hist.clear(); dyn_target=""
#                 else:
#                     dyn_count=0; dyn_target=""; dyn_hist.clear()
#             else:
#                 if not hands: prev_kp = None
#                 dyn_count=0; dyn_target=""; dyn_hist.clear()

#         # ── Static inference ─────────────────────────────────────────────────
#         stat_lbl = ""; stat_cf = 0.
#         if (static_model is not None and hands
#                 and not moving and dyn_count == 0):
#             sf = extract_static_features(res)
#             if not np.all(sf==0):
#                 si  = np.expand_dims(sf, 0)
#                 so  = static_model(si, training=False).numpy()[0]
#                 sbi = np.argmax(so)
#                 stat_cf  = so[sbi]
#                 stat_lbl = STATIC_ACTIONS[sbi]
#                 if stat_cf > STATIC_CONF_THRESHOLD:
#                     if stat_lbl == stat_target:
#                         stat_count += 1
#                     else:
#                         stat_target = stat_lbl
#                         stat_count  = 1
#                     if stat_count >= STATIC_STABLE_FRAMES:
#                         builder.add(stat_target, "sta")
#                         stat_count=0; stat_target=""
#                 else:
#                     stat_count=0; stat_target=""

#         # Key handling
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('t'):
#             c = builder.force()
#             if c: cur_sent = c
#         elif key == ord('c'):
#             builder.clear(); cur_sent = ""; print("  Cleared.")
#         elif key == ord('q'):
#             break

#         # ── UI ───────────────────────────────────────────────────────────────
#         h, w = img.shape[:2]

#         # Word buffer
#         cv2.rectangle(img, (0,0), (w,44), (40,40,40), -1)
#         wt = "Words: " + "  |  ".join(builder.words) if builder.words \
#              else "Words: (sign to begin...)"
#         cv2.putText(img, wt, (5,32),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

#         # Sentence history
#         n   = len(builder.history)
#         py  = 50
#         ph  = 28*(n+1)
#         cv2.rectangle(img, (0,py), (w,py+ph), (22,22,22), -1)
#         cv2.putText(img, "Sentences:", (5,py+18),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130,130,130), 1, cv2.LINE_AA)
#         for i, s in enumerate(builder.history):
#             b = int(130 + 125*((i+1)/max(n,1)))
#             cv2.putText(img, s, (5,py+18+(i+1)*26),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.60, (b,b,b), 2, cv2.LINE_AA)

#         # Status bar
#         cv2.rectangle(img, (0,h-88), (w,h), (28,28,28), -1)
#         sc = (0,220,80) if (moving and hands) else (80,140,220)
#         st = "SIGNING" if (moving and hands) else ("STILL" if hands else "NO HANDS")
#         cv2.putText(img, f"{st}  w:{we:.4f} a:{ae:.4f}",
#                     (5,h-62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, sc, 2, cv2.LINE_AA)

#         if dyn_lbl:
#             dc = (0,220,80) if dyn_cf>DYNAMIC_CONF_THRESHOLD else (0,165,255)
#             cv2.putText(img, f"DYN: {dyn_lbl}  {dyn_cf:.2f}",
#                         (5,h-36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, dc, 2, cv2.LINE_AA)

#         if stat_lbl:
#             cv2.putText(img, f"STA: {stat_lbl}  {stat_cf:.2f}",
#                         (5,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,100,255), 2, cv2.LINE_AA)

#         # Confirmation bar (green)
#         if dyn_count > 0 and dyn_target:
#             bw = int((dyn_count/STABLE_FRAMES)*180)
#             cv2.rectangle(img, (w-192,6),  (w-10,30),     (50,50,50),  -1)
#             cv2.rectangle(img, (w-192,6),  (w-192+bw,30), (0,200,100), -1)
#             cv2.putText(img, dyn_target[:22], (w-190,24),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,255,255), 1, cv2.LINE_AA)

#         # Static hold bar (purple)
#         if stat_count > 0 and stat_target:
#             bw = int((stat_count/STATIC_STABLE_FRAMES)*180)
#             cv2.rectangle(img, (w-192,34), (w-10,52),     (50,50,50),   -1)
#             cv2.rectangle(img, (w-192,34), (w-192+bw,52), (160,0,200),  -1)
#             cv2.putText(img, f"[S]{stat_target[:18]}", (w-190,48),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)

#         # Pause bar (blue)
#         if builder.no_hand > 0 and builder.words:
#             pp = min(builder.no_hand/PAUSE_FRAMES, 1.0)
#             cv2.rectangle(img, (w-192,56), (w-10,70),        (50,50,50),  -1)
#             cv2.rectangle(img, (w-192,56), (w-192+int(pp*182),70), (0,160,220), -1)
#             cv2.putText(img, "pause...", (w-190,66),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1, cv2.LINE_AA)

#         # LLM tag
#         cv2.putText(img, f"DYN:{os.path.basename(DYNAMIC_MODEL_FILE).split('_')[1][:3].upper()}"
#                          f"{'|STA' if static_model else ''}|LLM:Groq",
#                     (5,h-86), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,180,255), 1, cv2.LINE_AA)

#         cv2.imshow('ISL Translator', img)

# cap.release()
# cv2.destroyAllWindows()
"""
live_inference.py — Integrated ISL Real-Time Translator

Combines:
  1. Dynamic model  (TCN or Bi-LSTM) — motion-based signs, 720-dim features
  2. Static model   (Dense NN)       — static/fingerspell signs, 84-dim features
                                       uses the EXACT same feature extraction
                                       as create_dataset.py / inference_nn.py
  3. Groq LLM (Llama 3.3 70B)       — ISL word sequence → English sentence

Priority logic:
  - When hands are MOVING  → dynamic model runs, static model is idle
  - When hands are STILL   → static model runs, dynamic model is idle
  - Both feed into the same SentenceBuilder

Fixes applied v2:
  - STAT_STABLE reduced 8 → 4   (faster static detection)
  - WRIST_THRESH raised 0.004 → 0.006  (ignore tiny transition movements)
  - ANGLE_THRESH raised 0.002 → 0.003
  - STATIC_COOLDOWN = 1.5s: after a static sign is accepted, dynamic
    inference is fully suppressed so moving hand to next pose is not
    misread as a dynamic sign

Controls:
  T — force sentence completion → Groq
  C — clear everything
  Q — quit
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import time                             # ← ADDED
from collections import deque
from groq import Groq
import datetime
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Dynamic model (TCN / Bi-LSTM) ─────────────────────────────────────────────
DYNAMIC_MODEL_FILE = 'isl_tcn_model.h5'
DYNAMIC_CLASSES    = 'classes.npy'

# ── Static model (Dense NN) ───────────────────────────────────────────────────
STATIC_MODEL_FILE  = 'isl_landmark_model.h5'
STATIC_ENCODER     = 'label_encoder.pickle'

# ── Inference thresholds ───────────────────────────────────────────────────────
DYNAMIC_CONF  = 0.60
STATIC_CONF   = 0.85
WRIST_THRESH  = 0.006    # ← RAISED from 0.004 (ignore small transition movements)
ANGLE_THRESH  = 0.003    # ← RAISED from 0.002
DYN_STABLE    = 2
STAT_STABLE   = 4        # ← REDUCED from 8 (faster static detection)
INFER_EVERY   = 4

# ── Static → Dynamic transition suppression ───────────────────────────────────
STATIC_COOLDOWN = 1.5    # ← ADDED: seconds to suppress dynamic after static sign

# ── Sentence settings ──────────────────────────────────────────────────────────
PAUSE_FRAMES = 25
MAX_WORDS    = 8

# ── Words suppressed (false positive prone) ────────────────────────────────────
SUPPRESSED = {'it'}

# ── Time-of-day disambiguation ─────────────────────────────────────────────────
MORNING_CUTOFF = 12

# ── MediaPipe face indices (dynamic model) ─────────────────────────────────────
SELECTED_FACE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0,
    267, 269, 270, 409, 46, 53, 52, 65, 55, 285, 295, 282, 283, 276,
    33, 133, 362, 263, 1, 2, 98, 327, 152, 148
]

mp_holistic  = mp.solutions.holistic
mp_hands_mod = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils


# ==========================================
# 2. LOAD MODELS
# ==========================================
groq_client = Groq(api_key=GROQ_API_KEY)

print(f"Loading dynamic model : {DYNAMIC_MODEL_FILE}")
dynamic_model   = tf.keras.models.load_model(DYNAMIC_MODEL_FILE)
DYNAMIC_ACTIONS = np.load(DYNAMIC_CLASSES)
print(f"  {len(DYNAMIC_ACTIONS)} dynamic classes")

print(f"Loading static model  : {STATIC_MODEL_FILE}")
static_model = tf.keras.models.load_model(STATIC_MODEL_FILE, compile=False)
with open(STATIC_ENCODER, 'rb') as f:
    label_encoder = pickle.load(f)
STATIC_CLASSES = list(label_encoder.classes_)
print(f"  {len(STATIC_CLASSES)} static classes: {STATIC_CLASSES}")


# ==========================================
# 3. GROQ SENTENCE TRANSLATION
# ==========================================
def words_to_sentence(word_list):
    if not word_list:
        return ""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Indian Sign Language (ISL) interpreter. "
                        "ISL follows Subject-Object-Verb (SOV) word order. "
                        "Question signs appear at the end of the ISL sequence. "
                        "Convert the given ISL sign word sequence into one "
                        "natural English sentence. "
                        "Output only the sentence — no explanation, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"ISL signs in order: {', '.join(word_list)}"
                }
            ],
            max_tokens=120,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [Groq error] {e}")
        return _fallback(word_list)


def _fallback(words):
    if not words: return ""
    s  = words.copy()
    qw = {"what","where","who","how","why","when"}
    if s[-1].lower() in qw:
        s.insert(0, s.pop(-1))
    pr = {"i","you","he","she","we","they"}
    if len(s)==3 and s[0].lower() in pr:
        s[1], s[2] = s[2], s[1]
    out = " ".join(s).capitalize()
    return out + ("?" if any(w in out.lower() for w in qw) else ".")


# ==========================================
# 4. DYNAMIC MODEL FEATURE EXTRACTION
# ==========================================
def compute_joint_angles(hand_flat):
    if np.all(hand_flat == 0): return np.zeros(15)
    pts = hand_flat.reshape(21, 3)
    triplets = [
        (0,1,2),(1,2,3),
        (0,5,6),(5,6,7),(6,7,8),
        (0,9,10),(9,10,11),(10,11,12),
        (0,13,14),(13,14,15),(14,15,16),
        (0,17,18),(17,18,19),(18,19,20),
        (5,9,13),
    ]
    out = []
    for a, b, c in triplets:
        v1, v2 = pts[a]-pts[b], pts[c]-pts[b]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        out.append(0. if n1<1e-6 or n2<1e-6
                   else float(np.arccos(np.clip(
                       np.dot(v1,v2)/(n1*n2), -1., 1.))))
    return np.array(out)


def extract_dynamic_keypoints(results):
    pose = (np.array([[r.x,r.y,r.z]
                      for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(99))
    face = (np.array([[results.face_landmarks.landmark[i].x,
                       results.face_landmarks.landmark[i].y,
                       results.face_landmarks.landmark[i].z]
                      for i in SELECTED_FACE_INDICES]).flatten()
            if results.face_landmarks else np.zeros(120))
    lh = (np.array([[r.x,r.y,r.z]
                    for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(63))
    rh = (np.array([[r.x,r.y,r.z]
                    for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(63))
    raw = np.concatenate([pose, face, lh, rh])
    if results.pose_landmarks:
        nx = results.pose_landmarks.landmark[0].x
        ny = results.pose_landmarks.landmark[0].y
        nz = results.pose_landmarks.landmark[0].z
        raw[0::3] -= nx; raw[1::3] -= ny; raw[2::3] -= nz
    return raw


# ==========================================
# 5. STATIC MODEL FEATURE EXTRACTION
# ==========================================
def get_hand_vector(hand_landmarks):
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    min_x, min_y = min(x_), min(y_)
    temp = []
    for lm in hand_landmarks.landmark:
        temp.append(lm.x - min_x)
        temp.append(lm.y - min_y)
    max_val = max(abs(v) for v in temp)
    if max_val == 0:
        max_val = 1
    return [v / max_val for v in temp]


def extract_static_features(hand_results):
    left_hand_data  = [0.0] * 42
    right_hand_data = [0.0] * 42
    detected        = False
    if (hand_results.multi_hand_landmarks
            and hand_results.multi_handedness):
        for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
            label = hand_results.multi_handedness[i].classification[0].label
            vec   = get_hand_vector(hand_lm)
            if label == 'Left':
                left_hand_data  = vec
            else:
                right_hand_data = vec
            detected = True
    if not detected:
        return None
    return np.array(left_hand_data + right_hand_data, dtype=np.float32)


# ==========================================
# 6. MOTION GATE
# ==========================================
def motion_energy(sequence):
    seq     = np.array(sequence)
    wrist_e = np.mean(np.abs(np.diff(seq[:, 219:225], axis=0)))
    angle_e = np.mean(np.abs(np.diff(seq[:, 690:720], axis=0)))
    return wrist_e, angle_e


# ==========================================
# 7. TIME-BASED DISAMBIGUATION
# ==========================================
def disambiguate(label):
    hour  = datetime.datetime.now().hour
    clean = label.lower()
    if 'good' in clean and ('morning' in clean or 'afternoon' in clean):
        if hour < MORNING_CUTOFF:
            return label if 'morning' in clean else label.replace(
                'afternoon','morning').replace('Afternoon','Morning')
        else:
            return label if 'afternoon' in clean else label.replace(
                'morning','afternoon').replace('Morning','Afternoon')
    return label


# ==========================================
# 8. SENTENCE BUILDER
# ==========================================
class SentenceBuilder:
    def __init__(self):
        self.words   = []
        self.history = []
        self.no_hand = 0
        self.last    = ""

    def add(self, word, src="dyn"):
        clean = (''.join(c for c in word if not c.isdigit())
                 .replace('.','').strip().lower())
        if not clean:
            return
        if clean in SUPPRESSED:
            print(f"  [SUPPRESSED] '{clean}'")
            return
        if clean == self.last:
            return
        self.words.append(clean)
        self.last = clean
        print(f"  [{src.upper()}] '{clean}' → {self.words}")
        if len(self.words) >= MAX_WORDS:
            self._done()

    def tick(self, hands):
        if hands:
            self.no_hand = 0
            return None
        self.no_hand += 1
        if self.no_hand >= PAUSE_FRAMES and self.words:
            return self._done()
        return None

    def force(self):
        return self._done() if self.words else None

    def clear(self):
        self.words=[]; self.history=[]; self.no_hand=0; self.last=""

    def _done(self):
        print(f"\n  → Groq: {self.words}")
        s = words_to_sentence(self.words)
        self.history.append(s)
        if len(self.history) > 4:
            self.history = self.history[-4:]
        print(f"  ✓ {s}\n")
        self.words=[]; self.last=""; self.no_hand=0
        return s


# ==========================================
# 9. STATE
# ==========================================
sequence   = deque(maxlen=30)
prev_kp    = None
fc         = 0

dyn_target = ""; dyn_count = 0
dyn_hist   = deque(maxlen=DYN_STABLE)

stat_target      = ""
stat_count       = 0
last_static_time = 0.0    # ← ADDED: timestamp of last accepted static sign

builder  = SentenceBuilder()
cur_sent = ""


# ==========================================
# 10. MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam."); exit()

print(f"\nStarted.")
print(f"  Dynamic : {DYNAMIC_MODEL_FILE} ({len(DYNAMIC_ACTIONS)} classes)")
print(f"  Static  : {STATIC_MODEL_FILE}  ({len(STATIC_CLASSES)} classes)")
print(f"  LLM     : Groq / llama-3.3-70b-versatile")
print(f"  Static hold : {STAT_STABLE} frames  |  Cooldown : {STATIC_COOLDOWN}s")
print("  T=translate  C=clear  Q=quit\n")

hands_detector = mp_hands_mod.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        hol_res  = holistic.process(img_rgb)
        hand_res = hands_detector.process(img_rgb)

        img_rgb.flags.writeable = True
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if hol_res.face_landmarks:
            mp_drawing.draw_landmarks(img, hol_res.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10),  thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        if hol_res.pose_landmarks:
            mp_drawing.draw_landmarks(img, hol_res.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        if hol_res.left_hand_landmarks:
            mp_drawing.draw_landmarks(img, hol_res.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS)
        if hol_res.right_hand_landmarks:
            mp_drawing.draw_landmarks(img, hol_res.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS)

        # ── Dynamic feature extraction ────────────────────────────────────────
        kp  = extract_dynamic_keypoints(hol_res)
        vel = np.zeros(345) if prev_kp is None else kp - prev_kp
        prev_kp = kp.copy()

        lh_flat = (np.array([[r.x,r.y,r.z]
                              for r in hol_res.left_hand_landmarks.landmark]).flatten()
                   if hol_res.left_hand_landmarks else np.zeros(63))
        rh_flat = (np.array([[r.x,r.y,r.z]
                              for r in hol_res.right_hand_landmarks.landmark]).flatten()
                   if hol_res.right_hand_landmarks else np.zeros(63))

        combined = np.concatenate([kp, vel,
                                   compute_joint_angles(lh_flat),
                                   compute_joint_angles(rh_flat)])
        sequence.append(combined)
        fc += 1

        hands_vis = bool(hol_res.left_hand_landmarks or hol_res.right_hand_landmarks)
        we, ae    = motion_energy(sequence) if len(sequence)==30 else (0., 0.)
        moving    = we > WRIST_THRESH and ae > ANGLE_THRESH

        # Sentence builder pause detection
        completed = builder.tick(hands_vis)
        if completed: cur_sent = completed

        # ── ADDED: Cooldown check ─────────────────────────────────────────────
        in_static_cooldown = (time.time() - last_static_time) < STATIC_COOLDOWN

        # ── Dynamic inference ─────────────────────────────────────────────────
        # Fully suppressed during cooldown — transition movement ignored
        dyn_lbl = ""; dyn_cf = 0.
        if len(sequence)==30 and fc % INFER_EVERY == 0:
            if hands_vis and moving and not in_static_cooldown:  # ← ADDED check
                inp = np.expand_dims(np.array(sequence), 0)
                out = dynamic_model(inp, training=False).numpy()[0]
                bi  = np.argmax(out)
                dyn_cf  = float(out[bi])
                dyn_lbl = disambiguate(DYNAMIC_ACTIONS[bi])

                if dyn_cf > DYNAMIC_CONF:
                    if dyn_lbl == dyn_target:
                        dyn_count += 1
                        dyn_hist.append(dyn_cf)
                    else:
                        dyn_target = dyn_lbl
                        dyn_count  = 1
                        dyn_hist.clear()
                        dyn_hist.append(dyn_cf)

                    if (dyn_count >= DYN_STABLE and
                            np.mean(dyn_hist) > DYNAMIC_CONF):
                        builder.add(dyn_target, "dyn")
                        dyn_count=0; dyn_hist.clear(); dyn_target=""
                else:
                    dyn_count=0; dyn_target=""; dyn_hist.clear()
            else:
                if not hands_vis: prev_kp = None
                dyn_count=0; dyn_target=""; dyn_hist.clear()

        # ── Static inference ──────────────────────────────────────────────────
        stat_lbl = ""; stat_cf = 0.
        if hands_vis and not moving and dyn_count == 0:
            static_feat = extract_static_features(hand_res)
            if static_feat is not None:
                si  = np.expand_dims(static_feat, 0)
                so  = static_model(si, training=False).numpy()[0]
                sbi = np.argmax(so)
                stat_cf  = float(so[sbi])
                stat_lbl = label_encoder.inverse_transform([sbi])[0]

                if stat_cf > STATIC_CONF:
                    if stat_lbl == stat_target:
                        stat_count += 1
                    else:
                        stat_target = stat_lbl
                        stat_count  = 1

                    if stat_count >= STAT_STABLE:
                        builder.add(stat_target, "sta")
                        last_static_time = time.time()  # ← ADDED: start cooldown
                        stat_count=0; stat_target=""
                else:
                    stat_count=0; stat_target=""
            else:
                stat_count=0; stat_target=""

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            c = builder.force()
            if c: cur_sent = c
        elif key == ord('c'):
            builder.clear(); cur_sent=""
            last_static_time = 0.0   # ← ADDED: reset cooldown on clear
            print("  Cleared.")
        elif key == ord('q'):
            break

        # ── UI ────────────────────────────────────────────────────────────────
        h, w = img.shape[:2]

        # Word buffer bar
        cv2.rectangle(img, (0,0), (w,44), (40,40,40), -1)
        wt = ("Words: " + "  |  ".join(builder.words)
              if builder.words else "Words: (sign to begin...)")
        cv2.putText(img, wt, (5,32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        # Sentence history
        n  = len(builder.history)
        py = 50
        ph = 28*(n+1)
        cv2.rectangle(img, (0,py), (w,py+ph), (22,22,22), -1)
        cv2.putText(img, "Sentences:", (5,py+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130,130,130), 1, cv2.LINE_AA)
        for i, s in enumerate(builder.history):
            b = int(130 + 125*((i+1)/max(n,1)))
            cv2.putText(img, s, (5,py+18+(i+1)*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (b,b,b), 2, cv2.LINE_AA)

        # Status bar
        cv2.rectangle(img, (0,h-95), (w,h), (28,28,28), -1)

        # State label — shows cooldown countdown when active
        if in_static_cooldown:
            remaining   = STATIC_COOLDOWN - (time.time() - last_static_time)
            state_text  = f"COOLDOWN ({remaining:.1f}s) — dynamic suppressed"
            state_color = (0, 200, 255)
        elif moving and hands_vis:
            state_text  = "SIGNING (dynamic)"
            state_color = (0, 220, 80)
        elif hands_vis:
            state_text  = "STATIC HOLD"
            state_color = (80, 140, 220)
        else:
            state_text  = "NO HANDS"
            state_color = (100, 100, 100)

        cv2.putText(img, f"{state_text}  w:{we:.4f} a:{ae:.4f}",
                    (5,h-68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 2, cv2.LINE_AA)

        # Dynamic prediction
        if dyn_lbl:
            dc = (0,220,80) if dyn_cf>DYNAMIC_CONF else (0,165,255)
            cv2.putText(img, f"DYN: {dyn_lbl}  {dyn_cf:.2f}",
                        (5,h-42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, dc, 2, cv2.LINE_AA)

        # Static prediction
        if stat_lbl:
            sc2 = (0,220,80) if stat_cf>STATIC_CONF else (0,165,255)
            cv2.putText(img, f"STA: {stat_lbl}  {stat_cf:.2f}",
                        (5,h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, sc2, 2, cv2.LINE_AA)

        # Dynamic confirmation bar (green)
        if dyn_count > 0 and dyn_target:
            bw = int((dyn_count/DYN_STABLE)*180)
            cv2.rectangle(img, (w-192,6),  (w-10,28),     (50,50,50),  -1)
            cv2.rectangle(img, (w-192,6),  (w-192+bw,28), (0,200,100), -1)
            cv2.putText(img, dyn_target[:22], (w-190,22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)

        # Static hold bar (purple)
        if stat_count > 0 and stat_target:
            bw = int((stat_count/STAT_STABLE)*180)
            cv2.rectangle(img, (w-192,32), (w-10,52),     (50,50,50),   -1)
            cv2.rectangle(img, (w-192,32), (w-192+bw,52), (160,0,200),  -1)
            cv2.putText(img, f"[S]{stat_target[:18]}", (w-190,47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1, cv2.LINE_AA)

        # Cooldown bar (orange) ← ADDED
        if in_static_cooldown:
            elapsed  = time.time() - last_static_time
            progress = 1.0 - min(elapsed / STATIC_COOLDOWN, 1.0)
            bw = int(progress * 182)
            cv2.rectangle(img, (w-192,56), (w-10,70),         (50,50,50),  -1)
            cv2.rectangle(img, (w-192,56), (w-192+bw,70),     (0,165,255), -1)
            cv2.putText(img, "dyn suppressed", (w-190,66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 1, cv2.LINE_AA)

        # Pause bar (blue) — shifts down if cooldown bar is showing
        if builder.no_hand > 0 and builder.words:
            pp    = min(builder.no_hand/PAUSE_FRAMES, 1.0)
            bar_y = 74 if in_static_cooldown else 56
            cv2.rectangle(img, (w-192,bar_y), (w-10,bar_y+14),               (50,50,50),  -1)
            cv2.rectangle(img, (w-192,bar_y), (w-192+int(pp*182),bar_y+14),  (0,160,220), -1)
            cv2.putText(img, "pause...", (w-190,bar_y+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200,200,200), 1, cv2.LINE_AA)

        # Model indicator
        dyn_tag = os.path.basename(DYNAMIC_MODEL_FILE).split('_')[1][:3].upper()
        cv2.putText(img,
                    f"DYN:{dyn_tag} | STA:NN84 | LLM:Groq | hold:{STAT_STABLE}fr cd:{STATIC_COOLDOWN}s",
                    (5,h-93),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80,180,255), 1, cv2.LINE_AA)

        cv2.imshow('ISL Translator — Dynamic + Static + Groq', img)

hands_detector.close()
cap.release()
cv2.destroyAllWindows()