import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# ==========================================
# 1. SETUP & INITIALIZATION
# ==========================================
print("Loading model... (This takes a few seconds)")
model = tf.keras.models.load_model('isl_bilstm_model.h5') 

try:
    ACTIONS = np.load('classes.npy')
    print(f"Loaded {len(ACTIONS)} classes.")
except FileNotFoundError:
    ACTIONS = np.array(['dog', 'cat']) 
    print("Warning: classes.npy not found.")

SELECTED_FACE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 
    46, 53, 52, 65, 55, 285, 295, 282, 283, 276, 
    33, 133, 362, 263, 
    1, 2, 98, 327, 152, 148 
] 

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x, 
                          results.face_landmarks.landmark[i].y, 
                          results.face_landmarks.landmark[i].z] for i in SELECTED_FACE_INDICES]).flatten()
    else:
        face = np.zeros(40*3)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # --- PHASE 2: HAND WEIGHTING MULTIPLIER ---
    lh = lh * 2.0 
    rh = rh * 2.0 
    
    raw_features = np.concatenate([pose, face, lh, rh])
    
    # NORMALIZATION (Nose Anchor)
    if results.pose_landmarks:
        nose_x = results.pose_landmarks.landmark[0].x
        nose_y = results.pose_landmarks.landmark[0].y
        nose_z = results.pose_landmarks.landmark[0].z
        
        raw_features[0::3] -= nose_x 
        raw_features[1::3] -= nose_y 
        raw_features[2::3] -= nose_z 
        
    return raw_features

def apply_isl_grammar(sentence_buffer):
    if len(sentence_buffer) == 0:
        return ""
        
    sentence = sentence_buffer.copy()
    question_words = ["what", "where", "who", "how", "why"]
    
    if sentence[-1].lower() in question_words:
        q_word = sentence.pop(-1)
        sentence.insert(0, q_word)
        
    if len(sentence) == 3 and sentence[0].lower() in ['i', 'you', 'he', 'she']:
        sentence[1], sentence[2] = sentence[2], sentence[1]
        
    final_string = " ".join(sentence).capitalize()
    if any(q in final_string.lower() for q in question_words):
        final_string += "?"
    else:
        final_string += "."
        
    return final_string

# ==========================================
# 3. MAIN WEBCAM LOOP
# ==========================================
sequence = deque(maxlen=30) 
sentence_buffer = [] 
threshold = 0.85 

last_prediction = ""
current_prediction_target = ""
prediction_count = 0
stable_frames_required = 3 
frame_counter = 0 

cap = cv2.VideoCapture(0)
print("Starting Webcam...")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- PORTRAIT MODE OVERRIDE ---
        # If your phone is sideways to the left, use: cv2.ROTATE_90_CLOCKWISE
        # If it is sideways to the right, use: cv2.ROTATE_90_COUNTERCLOCKWISE
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        image, image.flags.writeable = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), False
        results = holistic.process(image)
        image.flags.writeable, image = True, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # --- DRAW LANDMARKS ---
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # --- DATA EXTRACTION (HYBRID) ---
        keypoints = extract_keypoints(results)
        
        if len(sequence) == 0:
            velocity = np.zeros(345)
        else:
            velocity = keypoints - sequence[-1][:345]
            
        combined_features = np.concatenate([keypoints, velocity])
        sequence.append(combined_features)
        
        frame_counter += 1 

        # --- INFERENCE LOGIC (PHASE 1 OPTIMIZED) ---
        hands_visible = results.left_hand_landmarks or results.right_hand_landmarks
        
        if len(sequence) == 30 and frame_counter % 5 == 0:
            if hands_visible:
                input_data = np.expand_dims(sequence, axis=0) 
                res = model(input_data, training=False).numpy()[0] 
                
                best_match_idx = np.argmax(res)
                
                if res[best_match_idx] > threshold: 
                    predicted_action = ACTIONS[best_match_idx]
                    
                    if predicted_action == current_prediction_target:
                        prediction_count += 1
                    else:
                        current_prediction_target = predicted_action
                        prediction_count = 1
                        
                    if prediction_count >= stable_frames_required:
                        if current_prediction_target != last_prediction:
                            clean_word = ''.join([i for i in current_prediction_target if not i.isdigit()]).replace('.', '').strip()
                            
                            sentence_buffer.append(clean_word)
                            last_prediction = current_prediction_target
                            prediction_count = 0 
            else:
                prediction_count = 0
                current_prediction_target = ""
                    
            if len(sentence_buffer) > 5: 
                sentence_buffer = sentence_buffer[-5:]
                
        # Press 'T' to Trigger Grammar Translation
        if cv2.waitKey(1) & 0xFF == ord('t'):
            english_sentence = apply_isl_grammar(sentence_buffer)
            print(f"\nISL Input: {sentence_buffer}")
            print(f"Translated English: {english_sentence}\n")
            sentence_buffer = [] 
            last_prediction = ""

        # UI Overlay (Adjusted width for portrait screen)
        cv2.rectangle(image, (0,0), (480, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence_buffer), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('ISL Real-Time Translator', image)

        # Press 'Q' to Quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()