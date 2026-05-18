import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import time
from collections import deque
from groq import Groq
import datetime
import os
import threading
import json
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# ==========================================
# 1. CONFIGURATION
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DYNAMIC_MODEL_FILE = 'isl_tcn_model.h5'
DYNAMIC_CLASSES    = 'classes.npy'
STATIC_MODEL_FILE  = 'isl_landmark_model.h5'
STATIC_ENCODER     = 'label_encoder.pickle'

DYNAMIC_CONF  = 0.60
STATIC_CONF   = 0.85
WRIST_THRESH  = 0.006
ANGLE_THRESH  = 0.003
DYN_STABLE    = 2
STAT_STABLE   = 4
INFER_EVERY   = 4
STATIC_COOLDOWN = 1.5
PAUSE_FRAMES = 25
MAX_WORDS    = 8
SUPPRESSED = {'it'}
MORNING_CUTOFF = 12

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
dynamic_model   = tf.keras.models.load_model(DYNAMIC_MODEL_FILE)
DYNAMIC_ACTIONS = np.load(DYNAMIC_CLASSES)
static_model = tf.keras.models.load_model(STATIC_MODEL_FILE, compile=False)
with open(STATIC_ENCODER, 'rb') as f:
    label_encoder = pickle.load(f)
STATIC_CLASSES = list(label_encoder.classes_)

# ==========================================
# 3. GLOBAL STATE FOR FRONTEND
# ==========================================
global_state = {
    "handsDetected": False,
    "moving": False,
    "wristE": 0.0,
    "angleE": 0.0,
    "dynOutput": {"label": "", "conf": 0.0},
    "statOutput": {"label": "", "conf": 0.0},
    "words": [],
    "history": [],
    "noHandFrames": 0
}

state_lock = threading.Lock()

# ==========================================
# 4. FUNCTIONS
# ==========================================
def words_to_sentence(word_list):
    if not word_list:
        return ""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an Indian Sign Language (ISL) interpreter. ISL follows Subject-Object-Verb (SOV) word order. Question signs appear at the end of the ISL sequence. Convert the given ISL sign word sequence into one natural English sentence. Output only the sentence — no explanation, nothing else."},
                {"role": "user", "content": f"ISL signs in order: {', '.join(word_list)}"}
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

def compute_joint_angles(hand_flat):
    if np.all(hand_flat == 0): return np.zeros(15)
    pts = hand_flat.reshape(21, 3)
    triplets = [(0,1,2),(1,2,3),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20),(5,9,13)]
    out = []
    for a, b, c in triplets:
        v1, v2 = pts[a]-pts[b], pts[c]-pts[b]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        out.append(0. if n1<1e-6 or n2<1e-6 else float(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1., 1.))))
    return np.array(out)

def extract_dynamic_keypoints(results):
    pose = (np.array([[r.x,r.y,r.z] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99))
    face = (np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in SELECTED_FACE_INDICES]).flatten() if results.face_landmarks else np.zeros(120))
    lh = (np.array([[r.x,r.y,r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63))
    rh = (np.array([[r.x,r.y,r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63))
    raw = np.concatenate([pose, face, lh, rh])
    if results.pose_landmarks:
        nx = results.pose_landmarks.landmark[0].x
        ny = results.pose_landmarks.landmark[0].y
        nz = results.pose_landmarks.landmark[0].z
        raw[0::3] -= nx; raw[1::3] -= ny; raw[2::3] -= nz
    return raw

def get_hand_vector(hand_landmarks):
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    min_x, min_y = min(x_), min(y_)
    temp = []
    for lm in hand_landmarks.landmark:
        temp.append(lm.x - min_x)
        temp.append(lm.y - min_y)
    max_val = max(abs(v) for v in temp)
    if max_val == 0: max_val = 1
    return [v / max_val for v in temp]

def extract_static_features(hand_results):
    left_hand_data  = [0.0] * 42
    right_hand_data = [0.0] * 42
    detected        = False
    if (hand_results.multi_hand_landmarks and hand_results.multi_handedness):
        for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
            label = hand_results.multi_handedness[i].classification[0].label
            vec   = get_hand_vector(hand_lm)
            if label == 'Left': left_hand_data  = vec
            else: right_hand_data = vec
            detected = True
    if not detected: return None
    return np.array(left_hand_data + right_hand_data, dtype=np.float32)

def motion_energy(sequence):
    seq     = np.array(sequence)
    wrist_e = np.mean(np.abs(np.diff(seq[:, 219:225], axis=0)))
    angle_e = np.mean(np.abs(np.diff(seq[:, 690:720], axis=0)))
    return wrist_e, angle_e

def disambiguate(label):
    hour  = datetime.datetime.now().hour
    clean = label.lower()
    if 'good' in clean and ('morning' in clean or 'afternoon' in clean):
        if hour < MORNING_CUTOFF:
            return label if 'morning' in clean else label.replace('afternoon','morning').replace('Afternoon','Morning')
        else:
            return label if 'afternoon' in clean else label.replace('morning','afternoon').replace('Morning','Afternoon')
    return label

builder_lock = threading.RLock()

class SentenceBuilder:
    def __init__(self):
        self.words   = []
        self.history = []
        self.no_hand = 0
        self.last    = ""
    def add(self, word, src="dyn"):
        with builder_lock:
            clean = (''.join(c for c in word if not c.isdigit()).replace('.','').strip().lower())
            if not clean or clean in SUPPRESSED or clean == self.last: return
            self.words.append(clean)
            self.last = clean
            self.sync_state()
            if len(self.words) >= MAX_WORDS:
                self.trigger_done()
    def tick(self, hands):
        with builder_lock:
            if hands:
                self.no_hand = 0
                self.sync_state()
                return
            self.no_hand += 1
            self.sync_state()
            if self.no_hand >= PAUSE_FRAMES and self.words:
                self.trigger_done()
    def force(self):
        with builder_lock:
            if self.words:
                self.trigger_done()
    def clear(self):
        with builder_lock:
            self.words=[]; self.no_hand=0; self.last=""
            self.sync_state()
    def trigger_done(self):
        with builder_lock:
            if not self.words: return
            words_copy = self.words.copy()
            self.words=[]; self.last=""; self.no_hand=0
            self.sync_state()
        threading.Thread(target=self._process_translation, args=(words_copy,)).start()
    def _process_translation(self, words_copy):
        s = words_to_sentence(words_copy)
        if s:
            with builder_lock:
                self.history.append({"words": words_copy, "sentence": s})
                if len(self.history) > 10: self.history = self.history[-10:]
                self.sync_state()
    def sync_state(self):
        with state_lock:
            global_state["words"] = self.words.copy()
            global_state["history"] = self.history.copy()
            global_state["noHandFrames"] = self.no_hand

builder = SentenceBuilder()

def generate_frames():
    cap = cv2.VideoCapture(0)
    sequence   = deque(maxlen=30)
    prev_kp    = None
    fc         = 0
    dyn_target = ""; dyn_count = 0
    dyn_hist   = deque(maxlen=DYN_STABLE)
    stat_target      = ""
    stat_count       = 0
    last_static_time = 0.0

    hands_detector = mp_hands_mod.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    while True:
        success, frame = cap.read()
        if not success: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        hol_res  = holistic.process(img_rgb)
        hand_res = hands_detector.process(img_rgb)

        img_rgb.flags.writeable = True
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if hol_res.face_landmarks: mp_drawing.draw_landmarks(img, hol_res.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        if hol_res.pose_landmarks: mp_drawing.draw_landmarks(img, hol_res.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        if hol_res.left_hand_landmarks: mp_drawing.draw_landmarks(img, hol_res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if hol_res.right_hand_landmarks: mp_drawing.draw_landmarks(img, hol_res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        kp  = extract_dynamic_keypoints(hol_res)
        vel = np.zeros(345) if prev_kp is None else kp - prev_kp
        prev_kp = kp.copy()

        lh_flat = (np.array([[r.x,r.y,r.z] for r in hol_res.left_hand_landmarks.landmark]).flatten() if hol_res.left_hand_landmarks else np.zeros(63))
        rh_flat = (np.array([[r.x,r.y,r.z] for r in hol_res.right_hand_landmarks.landmark]).flatten() if hol_res.right_hand_landmarks else np.zeros(63))

        combined = np.concatenate([kp, vel, compute_joint_angles(lh_flat), compute_joint_angles(rh_flat)])
        sequence.append(combined)
        fc += 1

        hands_vis = bool(hol_res.left_hand_landmarks or hol_res.right_hand_landmarks)
        we, ae    = motion_energy(sequence) if len(sequence)==30 else (0., 0.)
        moving    = we > WRIST_THRESH and ae > ANGLE_THRESH

        builder.tick(hands_vis)
        in_static_cooldown = (time.time() - last_static_time) < STATIC_COOLDOWN

        dyn_lbl = ""; dyn_cf = 0.
        if len(sequence)==30 and fc % INFER_EVERY == 0:
            if hands_vis and moving and not in_static_cooldown:
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

                    if (dyn_count >= DYN_STABLE and np.mean(dyn_hist) > DYNAMIC_CONF):
                        builder.add(dyn_target, "dyn")
                        dyn_count=0; dyn_hist.clear(); dyn_target=""
                else:
                    dyn_count=0; dyn_target=""; dyn_hist.clear()
            else:
                if not hands_vis: prev_kp = None
                dyn_count=0; dyn_target=""; dyn_hist.clear()

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
                        last_static_time = time.time()
                        stat_count=0; stat_target=""
                else:
                    stat_count=0; stat_target=""
            else:
                stat_count=0; stat_target=""

        # Update global state
        with state_lock:
            global_state["handsDetected"] = bool(hands_vis)
            global_state["moving"] = bool(moving)
            global_state["wristE"] = float(we)
            global_state["angleE"] = float(ae)
            global_state["dynOutput"] = {"label": dyn_lbl, "conf": dyn_cf}
            global_state["statOutput"] = {"label": stat_lbl, "conf": stat_cf}

        ret, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# 5. FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('frontend', path)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def sse_state():
    def event_stream():
        while True:
            with state_lock:
                state_copy = global_state.copy()
            yield f"data: {json.dumps(state_copy)}\n\n"
            time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/force_translate', methods=['POST'])
def force_translate():
    builder.force()
    return jsonify({"status": "success"})

@app.route('/clear', methods=['POST'])
def clear():
    builder.clear()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
